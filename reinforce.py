from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from torch.nn.functional import log_softmax
from torch.optim import Adam

from cartpole import CartPoleConfig
from models import CartPolePolicy, parameterize_state
from rl import RewardMetricConfig, run_rollout

load_dotenv()

ARTIFACT_PATH = Path("./.artifacts/reinforce.pt")
ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

pi = CartPolePolicy()

env_cfg = CartPoleConfig()
rwd_cfg = RewardMetricConfig()

N_ROLLOUTS = 100
RETURN_GAMMA = 0.99
LR = 0.0001
ENTROPY_COEF = 0.01

n_env_steps = 0

opt = Adam(pi.parameters(), lr=LR)

wandb.init(
    config={
        "algorithm": "reinforce",
        "n_rollouts": N_ROLLOUTS,
        "return_gamma": RETURN_GAMMA,
        "lr": LR,
        "entropy_coef": ENTROPY_COEF,
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

    G_all = torch.cat(returns)
    mean, std = G_all.mean(), G_all.std()
    returns = [(g - mean) for g in returns]

    rollout_lengths = [len(rollout) for rollout in rollouts]
    avg_rollout_length = sum(rollout_lengths) / len(rollout_lengths)
    avg_alive_time = avg_rollout_length * env_cfg.dt

    reward_log_policy = 0
    entropies: list[torch.Tensor] = []

    for i, rollout in enumerate(rollouts):
        # G = (G_0, ..., G_t)
        G = returns[i]

        # a = (a_0, ..., a_t)
        a = torch.tensor([frame.action for frame in rollout])

        # S: (t, num_state_features)
        S = torch.stack([parameterize_state(frame.state) for frame in rollout])

        # P: (t, num_actions)
        P = log_softmax(pi(S), dim=-1)

        # p: (t,) = (log pi(a | s), ...)
        p = P.take_along_dim(a.unsqueeze(-1), 1).squeeze(-1)

        reward_log_policy += (G * p).sum()
        entropies.append(-(P.exp() * P).sum(-1))

    mean_entropy = torch.cat(entropies).mean()
    avg_entropy = mean_entropy.item()

    # optimizer step
    opt.zero_grad()
    loss = -reward_log_policy / N_ROLLOUTS - ENTROPY_COEF * mean_entropy
    loss.backward()
    opt.step()

    n_env_steps += sum(rollout_lengths)
    print("avg_time_alive", avg_alive_time)

    wandb.log(
        {
            "n_env_steps": n_env_steps,
            "loss": loss.item(),
            "reward_log_policy": reward_log_policy.item(),
            "avg_alive_time": avg_alive_time,
            "return_mean": mean.item(),
            "return_std": std.item(),
            "avg_entropy": avg_entropy,
        }
    )

    torch.save(pi.state_dict(), ARTIFACT_PATH)
