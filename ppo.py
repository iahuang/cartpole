from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from torch.nn.functional import log_softmax
from torch.optim import Adam

from cartpole import CartPoleConfig
from models import (
    ACTION_DIM,
    STATE_FEATURE_DIM,
    CartPolePolicy,
    CartPoleValue,
    parameterize_state,
)
from rl import CartPoleRewardConfig, RunningMeanStd, run_rollout

load_dotenv()

ARTIFACT_PATH = Path("./.artifacts/ppo.pt")
ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

pi = CartPolePolicy()
critic = CartPoleValue()

env_cfg = CartPoleConfig()
rwd_cfg = CartPoleRewardConfig(w_rel_distance=0.5, action_change_penalty=0.1)

ROLLOUT_BUFFER_SIZE = 2048

RETURN_GAMMA = 0.99
GAE_LAMBDA = 0.95
W_CRITIC_LOSS = 0.5
W_ENTROPY = 0.01
PI_LR = 3e-4
CRITIC_LR = 3e-4
MAX_GRAD_NORM = 1.0

CLIP_EPS = 0.2

UPDATE_EPOCHS = 4
UPDATE_BATCH_SIZE = 64

n_env_steps = 0


wandb.init(
    config={
        "algorithm": "ppo",
        "rollout_buffer_size": ROLLOUT_BUFFER_SIZE,
        "return_gamma": RETURN_GAMMA,
        "pi_lr": PI_LR,
        "critic_lr": CRITIC_LR,
        "dt": env_cfg.dt,
    },
    dir=".wandb",
)
wandb.define_metric("n_env_steps")
wandb.define_metric("*", step_metric="n_env_steps")


@dataclass
class RolloutBuffer:
    # (B,)
    rewards: torch.Tensor
    # (B, ACTION_DIM)
    logprobs: torch.Tensor
    # (B,)
    actions: torch.Tensor
    # (B,)
    action_idx: torch.Tensor
    # (B, STATE_FEATURE_DIM)
    states: torch.Tensor
    episode_lengths: list[int]
    # True iff the final episode in the buffer was truncated by the buffer
    # cap (i.e. did not reach a terminal state). When True, `final_state`
    # holds s_T for bootstrapping; otherwise the last episode ended in a
    # true terminal.
    last_truncated: bool
    # (STATE_FEATURE_DIM,) — state right after the buffer's last step.
    final_state: torch.Tensor


def build_rollout_buffer(
    buffer_size: int,
    pi: CartPolePolicy,
    env_config: CartPoleConfig,
    rwd_config: CartPoleRewardConfig,
) -> RolloutBuffer:
    """
    Computes a rollout buffer of the given number of steps.
    """

    rewards = torch.zeros(buffer_size, dtype=torch.float32)
    actions = torch.zeros(buffer_size, dtype=torch.int64)
    action_idx = torch.zeros(buffer_size, dtype=torch.int64)
    states = torch.zeros((buffer_size, STATE_FEATURE_DIM), dtype=torch.float32)
    logits = torch.zeros((buffer_size, ACTION_DIM), dtype=torch.float32)

    episode_lengths: list[int] = []
    last_truncated = False
    final_state = torch.zeros(STATE_FEATURE_DIM, dtype=torch.float32)

    i = 0
    n_episodes = 0

    with torch.no_grad():
        while i < buffer_size:
            rollout, terminated, end_state = run_rollout(
                pi,
                env_config=env_config,
                rwd_config=rwd_config,
                max_steps=buffer_size - i,
            )

            for t in range(len(rollout)):
                rewards[i + t] = rollout[t].reward
                actions[i + t] = rollout[t].action
                action_idx[i + t] = rollout[t].action_idx
                states[i + t] = parameterize_state(rollout[t].state)
                logits[i + t] = rollout[t].logits

            n_episodes += 1
            episode_lengths.append(len(rollout))
            i += len(rollout)

            # Only the most recent rollout's status matters at loop exit.
            last_truncated = not terminated
            final_state = parameterize_state(end_state)

        return RolloutBuffer(
            rewards=rewards,
            actions=actions,
            action_idx=action_idx,
            states=states,
            episode_lengths=episode_lengths,
            last_truncated=last_truncated,
            final_state=final_state,
            logprobs=log_softmax(logits, dim=-1),
        )


def compute_gae(
    buffer: RolloutBuffer,
    values: torch.Tensor,
    gamma: float,
    *,
    last_value: float | torch.Tensor = 0.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    buffer:     RolloutBuffer containing the rollout data
    values:     (T,)  V(s_t) — saved from the rollout's forward passes
    last_value: V(s_T) bootstrap. Use 0 if the rollout ended on a terminal,
                else the critic's value at the final next-state.
    """

    dones = torch.zeros_like(buffer.rewards)
    i = 0
    last_ep_idx = len(buffer.episode_lengths) - 1
    for ep_idx, length in enumerate(buffer.episode_lengths):
        # The final episode may have been truncated by the buffer cap rather than
        # ending in a true terminal state; in that case the bootstrap is handled
        # via `last_value` instead of zeroing it out here.
        if not (ep_idx == last_ep_idx and buffer.last_truncated):
            dones[i + length - 1] = 1.0
        i += length

    T = len(buffer.rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = buffer.rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


pi_optim = torch.optim.Adam(pi.parameters(), lr=PI_LR)
critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

n_env_steps = 0

return_rms = RunningMeanStd()

best_avg_time_alive = float("-inf")

while True:
    buffer = build_rollout_buffer(ROLLOUT_BUFFER_SIZE, pi, env_cfg, rwd_cfg)

    avg_time_alive = sum(buffer.episode_lengths) / len(buffer.episode_lengths)
    if avg_time_alive >= best_avg_time_alive:
        best_avg_time_alive = avg_time_alive
        torch.save(pi.state_dict(), ARTIFACT_PATH)

    with torch.no_grad():
        values_unnorm = critic(buffer.states) * return_rms.std + return_rms.mean
        if buffer.last_truncated:
            final_value_unnorm = (
                critic(buffer.final_state) * return_rms.std + return_rms.mean
            )
            last_value = final_value_unnorm
        else:
            last_value = 0.0

    advantages, returns = compute_gae(
        buffer, values_unnorm, RETURN_GAMMA, last_value=last_value, lam=GAE_LAMBDA
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_logprobs = buffer.logprobs.gather(1, buffer.action_idx.unsqueeze(-1)).squeeze(
        -1
    )

    n_env_steps += ROLLOUT_BUFFER_SIZE
    return_rms.update(returns)
    normalized_returns = (returns - return_rms.mean) / (return_rms.std + 1e-8)

    for epoch in range(UPDATE_EPOCHS):
        indices = torch.randperm(ROLLOUT_BUFFER_SIZE)
        epoch_kls = []

        for start in range(0, ROLLOUT_BUFFER_SIZE, UPDATE_BATCH_SIZE):
            mb_idx = indices[start : start + UPDATE_BATCH_SIZE]

            mb_states = buffer.states[mb_idx]
            mb_actions = buffer.action_idx[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_old_logprobs_all = buffer.logprobs[mb_idx]
            mb_advantages = advantages[mb_idx]

            # pi(s_t)
            curr_logprobs_all = log_softmax(pi(mb_states), dim=-1)
            curr_logprobs = curr_logprobs_all.gather(
                1, mb_actions.unsqueeze(-1)
            ).squeeze(-1)

            # r_phi
            policy_ratio = torch.exp(curr_logprobs - mb_old_logprobs)

            loss_clip = torch.min(
                policy_ratio * mb_advantages,
                torch.clamp(policy_ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages,
            )
            loss_clip = loss_clip.mean()

            loss_critic = (critic(mb_states) - normalized_returns[mb_idx]) ** 2
            loss_critic = loss_critic.mean()

            entropy = -(curr_logprobs_all.exp() * curr_logprobs_all).sum(-1).mean()

            with torch.no_grad():
                kl = (
                    (
                        mb_old_logprobs_all.exp()
                        * (mb_old_logprobs_all - curr_logprobs_all)
                    )
                    .sum(-1)
                    .mean()
                )

            loss = -loss_clip + W_CRITIC_LOSS * loss_critic - W_ENTROPY * entropy

            pi_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()

            grad_norm_pi = torch.nn.utils.clip_grad_norm_(
                pi.parameters(), MAX_GRAD_NORM
            )
            grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), MAX_GRAD_NORM
            )

            pi_optim.step()
            critic_optim.step()

            print(
                f"n_env_steps: {n_env_steps:06d}, loss: {loss.item():.4f}, avg_time_alive: {sum(buffer.episode_lengths) / len(buffer.episode_lengths):.2f} mean_return: {returns.mean():.2f}"
            )

            with torch.no_grad():
                explained_var = 1.0 - (returns - values_unnorm).var() / (
                    returns.var() + 1e-8
                )
                clip_frac = ((policy_ratio - 1.0).abs() > CLIP_EPS).float().mean()

            epoch_kls.append(kl.item())

            wandb.log(
                {
                    "grad_norm_pi": grad_norm_pi.item(),
                    "grad_norm_critic": grad_norm_critic.item(),
                    "loss_clip": loss_clip.item(),
                    "loss_critic": loss_critic.item(),
                    "entropy": entropy.item(),
                    "kl": kl.item(),
                    "loss": loss.item(),
                    "avg_time_alive": (
                        sum(buffer.episode_lengths) / len(buffer.episode_lengths)
                    )
                    * env_cfg.dt,
                    "n_env_steps": n_env_steps,
                    "explained_var": explained_var.item(),
                    "clip_frac": clip_frac.item(),
                }
            )

        if sum(epoch_kls) / len(epoch_kls) > 0.02:
            print(
                f"Early stopping at epoch {epoch} ({sum(epoch_kls) / len(epoch_kls):.4f} kl)"
            )
            break
