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

ARTIFACT_PATH = Path("./.artifacts/ppo_swingup.pt")
ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

pi = CartPolePolicy()
critic = CartPoleValue()

env_cfg = CartPoleConfig(ruleset="swing_up")
rwd_cfg = CartPoleRewardConfig(
    w_rel_distance=0.5,
    action_change_penalty=0.1,
    w_time_alive=2.0,
    w_cosine_angle=1.0,
    w_energy_target_penalty=0.3,
)

ROLLOUT_BUFFER_SIZE = 512 * 16
MAX_EPISODE_STEPS = 512

RETURN_GAMMA = 0.999
GAE_LAMBDA = 0.95
W_CRITIC_LOSS = 0.5
W_ENTROPY = 0.1
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
class EpisodeInfo:
    length: int
    # True iff the episode reached a natural terminal state (cart left the
    # x_threshold). False iff it was cut off by the per-episode step cap or
    # the buffer-fill cap — in that case the policy's expected future value
    # at `end_state` should be bootstrapped into the final reward.
    terminated: bool
    # (STATE_FEATURE_DIM,) — s_T, the state right after the episode's last step.
    end_state: torch.Tensor


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
    episodes: list[EpisodeInfo]


def build_rollout_buffer(
    buffer_size: int,
    pi: CartPolePolicy,
    env_config: CartPoleConfig,
    rwd_config: CartPoleRewardConfig,
    max_episode_steps: int,
) -> RolloutBuffer:
    """
    Computes a rollout buffer of the given number of steps. No episode runs
    longer than `max_episode_steps` — the simulator is reset and a fresh
    episode begins once the cap is hit.
    """

    rewards = torch.zeros(buffer_size, dtype=torch.float32)
    actions = torch.zeros(buffer_size, dtype=torch.int64)
    action_idx = torch.zeros(buffer_size, dtype=torch.int64)
    states = torch.zeros((buffer_size, STATE_FEATURE_DIM), dtype=torch.float32)
    logits = torch.zeros((buffer_size, ACTION_DIM), dtype=torch.float32)

    episodes: list[EpisodeInfo] = []

    i = 0

    with torch.no_grad():
        while i < buffer_size:
            max_steps = min(max_episode_steps, buffer_size - i)
            rollout, terminated, end_state = run_rollout(
                pi,
                env_config=env_config,
                rwd_config=rwd_config,
                max_steps=max_steps,
            )

            for t in range(len(rollout)):
                rewards[i + t] = rollout[t].reward
                actions[i + t] = rollout[t].action
                action_idx[i + t] = rollout[t].action_idx
                states[i + t] = parameterize_state(rollout[t].state)
                logits[i + t] = rollout[t].logits

            episodes.append(
                EpisodeInfo(
                    length=len(rollout),
                    terminated=terminated,
                    end_state=parameterize_state(end_state),
                )
            )
            i += len(rollout)

        return RolloutBuffer(
            rewards=rewards,
            actions=actions,
            action_idx=action_idx,
            states=states,
            episodes=episodes,
            logprobs=log_softmax(logits, dim=-1),
        )


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    *,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    rewards: (T,)  immediate rewards. For steps where an episode was truncated
             (not naturally terminated), the caller should pre-bake the
             discounted bootstrap value γ·V(s_T) into rewards[t].
    dones:   (T,)  1 at the last step of every episode (terminal OR truncated),
             0 elsewhere. Both reset the GAE recursion across boundaries.
    values:  (T,)  V(s_t) at each buffer state.
    """

    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


pi_optim = torch.optim.Adam(pi.parameters(), lr=PI_LR)
critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

n_env_steps = 0

return_rms = RunningMeanStd()

best_mean_return = float("-inf")

while True:
    buffer = build_rollout_buffer(
        ROLLOUT_BUFFER_SIZE, pi, env_cfg, rwd_cfg, MAX_EPISODE_STEPS
    )

    avg_time_alive = sum(e.length for e in buffer.episodes) / len(buffer.episodes)

    # Undiscounted per-episode return, averaged across episodes in the buffer.
    # Computed off `buffer.rewards` (pre-bootstrap), so it reflects only what
    # the environment actually paid out.
    ep_returns = []
    offset = 0
    for ep in buffer.episodes:
        ep_returns.append(
            float(buffer.rewards[offset : offset + ep.length].sum().item())
        )
        offset += ep.length
    mean_ep_return = sum(ep_returns) / len(ep_returns)

    if mean_ep_return >= best_mean_return:
        best_mean_return = mean_ep_return
        torch.save(pi.state_dict(), ARTIFACT_PATH)

    with torch.no_grad():
        values_unnorm = critic(buffer.states) * return_rms.std + return_rms.mean

        # Build the done mask and collect end-states of truncated (non-terminal)
        # episodes so we can bootstrap γ·V(s_T) into their final reward.
        dones = torch.zeros_like(buffer.rewards)
        truncated_indices: list[int] = []
        truncated_end_states: list[torch.Tensor] = []
        offset = 0
        for ep in buffer.episodes:
            end_idx = offset + ep.length - 1
            dones[end_idx] = 1.0
            if not ep.terminated:
                truncated_indices.append(end_idx)
                truncated_end_states.append(ep.end_state)
            offset += ep.length

        effective_rewards = buffer.rewards.clone()
        if truncated_end_states:
            stacked = torch.stack(truncated_end_states)
            bootstrap_values = critic(stacked) * return_rms.std + return_rms.mean
            for k, idx in enumerate(truncated_indices):
                effective_rewards[idx] = (
                    effective_rewards[idx] + RETURN_GAMMA * bootstrap_values[k]
                )

    advantages, returns = compute_gae(
        effective_rewards,
        dones,
        values_unnorm,
        RETURN_GAMMA,
        lam=GAE_LAMBDA,
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
                f"n_env_steps: {n_env_steps:06d}, loss: {loss.item():.4f}, avg_time_alive: {avg_time_alive:.2f} mean_return: {mean_ep_return:.2f}"
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
                    "avg_time_alive": avg_time_alive * env_cfg.dt,
                    "mean_return": mean_ep_return,
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
