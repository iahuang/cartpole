from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from torch.nn.functional import log_softmax
from torch.optim import Adam

from cartpole import CartPoleConfig
from models import CartPolePolicyWithValue, parameterize_state
from rl import CartPoleRewardConfig, run_rollout

load_dotenv()

ARTIFACT_PATH = Path("./.artifacts/reinforce.pt")
ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

pi = CartPolePolicyWithValue()

env_cfg = CartPoleConfig()
rwd_cfg = CartPoleRewardConfig()

N_ROLLOUTS = 100
RETURN_GAMMA = 0.99
W_CRITIC_LOSS = 0.5
W_ENTROPY_LOSS = 0.01
LR = 0.0001


n_env_steps = 0

opt = Adam(pi.parameters(), lr=LR)

wandb.init(
    config={
        "algorithm": "reinforce",
        "n_rollouts": N_ROLLOUTS,
        "return_gamma": RETURN_GAMMA,
        "lr": LR,
        "entropy_coef": W_ENTROPY_LOSS,
        "dt": env_cfg.dt,
    },
    dir=".wandb",
)
wandb.define_metric("n_env_steps")
wandb.define_metric("*", step_metric="n_env_steps")

while True:
    rollouts = [
        run_rollout(
            pi,
            env_config=env_cfg,
            rwd_config=rwd_cfg,
        )
        for _ in range(N_ROLLOUTS)
    ]

    returns: list[torch.Tensor] = []

    # compute G_t for all tau
    for i, rollout in enumerate(rollouts):
        r_returns = []
        G = 0

        for t in range(len(rollout) - 1, -1, -1):
            G = rollout[t].reward + RETURN_GAMMA * G
            r_returns.append(G)

        r_returns.reverse()
        returns.append(torch.tensor(r_returns))

    rollout_lengths = [len(rollout) for rollout in rollouts]
    avg_rollout_length = sum(rollout_lengths) / len(rollout_lengths)
    avg_alive_time = avg_rollout_length * env_cfg.dt

    # normalize returns globally so critic targets and advantages are unit-scale
    G_all = torch.cat(returns)
    G_mean, G_std = G_all.mean(), G_all.std()
    returns = [(g - G_mean) / (G_std + 1e-8) for g in returns]

    entropies: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    value_sum_sqerror = 0

    for i, rollout in enumerate(rollouts):
        # G = (G_0, ..., G_t) (normalized)
        G = returns[i]

        # a = (a_0, ..., a_t)
        a = torch.tensor([frame.action for frame in rollout])

        # S: (t, num_state_features)
        S = torch.stack([parameterize_state(frame.state) for frame in rollout])

        # pi_S: (t, num_state_features)
        # v_S: (t, 1)
        pi_S, v_S = pi(S)
        # P: (t, num_actions)
        P = log_softmax(pi_S, dim=-1)

        # p: (t,) = (log pi(a | s), ...)
        p = P.take_along_dim(a.unsqueeze(-1), 1).squeeze(-1)

        advantages.append((G - v_S.squeeze(-1)).detach())
        log_probs.append(p)
        entropies.append(-(P.exp() * P).sum(-1))
        value_sum_sqerror += ((v_S.squeeze(-1) - G) ** 2).sum(-1)

    all_advantages = torch.cat(advantages)
    all_log_probs = torch.cat(log_probs)
    norm_advantages = (all_advantages - all_advantages.mean()) / (
        all_advantages.std() + 1e-8
    )

    mean_entropy = torch.cat(entropies).mean()
    mean_value_sq_error = value_sum_sqerror / sum(rollout_lengths)
    mean_policy_reward = (norm_advantages * all_log_probs).mean()

    loss_policy = -mean_policy_reward
    loss_entropy = -W_ENTROPY_LOSS * mean_entropy
    loss_value = W_CRITIC_LOSS * mean_value_sq_error
    loss = loss_policy + loss_entropy + loss_value

    # optimizer step
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(pi.parameters(), float("inf"))
    opt.step()

    n_env_steps += sum(rollout_lengths)
    print("avg_time_alive", avg_alive_time)

    wandb.log(
        {
            "n_env_steps": n_env_steps,
            "loss": loss.item(),
            "loss_policy": loss_policy.item(),
            "loss_entropy": loss_entropy.item(),
            "loss_value": loss_value.item(),
            "avg_alive_time": avg_alive_time,
            "return_mean": G_mean.item(),
            "return_std": G_std.item(),
            "avg_entropy": mean_entropy.item(),
            "grad_norm": grad_norm.item(),
        }
    )

    torch.save(pi.state_dict(), ARTIFACT_PATH)
