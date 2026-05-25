import math
from dataclasses import dataclass

import torch

from cartpole import CartPole, CartPoleConfig, CartPoleState
from models import CartPolePolicy


@dataclass
class RolloutEntry:
    state: CartPoleState
    action: int
    action_idx: int
    reward: float
    logits: torch.Tensor


Rollout = list[RolloutEntry]


@dataclass
class CartPoleRewardConfig:
    # coefficient for number of timesteps alive. should be positive.
    w_time_alive: float = 1.0

    # coefficient for cos(theta). should be positive.
    w_cosine_angle: float = 0.0

    # coefficient for (|x| / x_threshold). should be negative.
    w_rel_distance: float = 0.0

    # penalty coefficient on (a_t - a_{t-1})^2 — subtracts c * Δa^2 from
    # each step's reward to discourage rapid action changes. Should be
    # non-negative. No penalty is applied on the very first step.
    action_change_penalty: float = 0.0


def run_rollout(
    pi: CartPolePolicy,
    *,
    env_config: CartPoleConfig,
    rwd_config: CartPoleRewardConfig,
    greedy: bool = False,
    max_steps: int | None = None,
) -> tuple[Rollout, bool, CartPoleState]:
    """
    Simulate a rollout of the cartpole environment using the given policy.

    Args:
        pi: The policy to use for selecting actions.
        env_config: The configuration for the cartpole environment.
        rwd_config: The configuration for the reward metric.
        greedy: Whether to use greedy mode (temperature = 0).

    Returns:
        A tuple of (rollout, terminated, final_state):
          - rollout: list of RolloutEntry. Each entry's `state` is s_t (the
            state from which `action` was sampled), so state/action/logits
            are time-aligned.
          - terminated: True if the episode reached a terminal state, False
            if it was truncated by max_steps.
          - final_state: env.state after the last step (s_T). For terminated
            episodes this is the terminal state; for truncated episodes it
            is the next state to bootstrap from.
    """

    env = CartPole(config=env_config)

    rollout: Rollout = []
    prev_action: int | None = None

    with torch.no_grad():
        n_env_steps = 0

        while True:
            s_t = env.state
            action, action_idx, logits = pi.sample(
                s_t, temperature=0.0 if greedy else 1.0
            )
            next_state, done = env.step(action)
            delta_a = 0 if prev_action is None else (action - prev_action)
            reward = (
                rwd_config.w_time_alive
                + rwd_config.w_cosine_angle * math.cos(next_state.theta)
                + rwd_config.w_rel_distance
                * (1 - abs(next_state.x) / env_config.x_threshold)
                - rwd_config.action_change_penalty * (delta_a * delta_a)
            )
            rollout.append(
                RolloutEntry(
                    state=s_t,
                    action=action,
                    action_idx=action_idx,
                    reward=reward,
                    logits=logits,
                )
            )
            prev_action = action
            if done:
                return rollout, True, next_state
            n_env_steps += 1
            if max_steps is not None and n_env_steps >= max_steps:
                return rollout, False, next_state


def compute_returns(rollout: Rollout, gamma: float) -> torch.Tensor:
    """
    Compute the returns for each step in the rollout using the given gamma.
    """

    r_returns: list[float] = []
    G = 0.0

    for t in range(len(rollout) - 1, -1, -1):
        G = rollout[t].reward + gamma * G
        r_returns.append(G)

    r_returns.reverse()
    return torch.tensor(r_returns)


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon  # avoids div-by-zero on first update

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a, m_b = self.var * self.count, batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    @property
    def std(self):
        return torch.sqrt(self.var)
