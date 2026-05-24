import math
from dataclasses import dataclass

import torch

from cartpole import CartPole, CartPoleConfig, CartPoleState
from models import CartPolePolicy


@dataclass
class RolloutEntry:
    state: CartPoleState
    action: int
    reward: float


Rollout = list[RolloutEntry]


@dataclass
class RewardMetricConfig:
    # coefficient for number of timesteps alive. should be positive.
    w_time_alive: float = 1.0

    # coefficient for cos(theta). should be positive.
    w_cosine_angle: float = 0.0

    # coefficient for (|x| / x_threshold). should be negative.
    w_rel_distance: float = 0.0


def run_rollout(
    pi: CartPolePolicy,
    *,
    env_config: CartPoleConfig,
    rwd_config: RewardMetricConfig,
    greedy: bool = False,
) -> Rollout:
    """
    Simulate a rollout of the cartpole environment using the given policy.

    Args:
        pi: The policy to use for selecting actions.
        env_config: The configuration for the cartpole environment.
        rwd_config: The configuration for the reward metric.
        greedy: Whether to use greedy mode (temperature = 0).

    Returns:
        A tuple containing the rollout.
    """

    env = CartPole(config=env_config)

    rollout: Rollout = []

    with torch.no_grad():
        while True:
            action = pi.sample(env.state, temperature=0.0 if greedy else 1.0)
            state, done = env.step(action)
            reward = (
                rwd_config.w_time_alive * env.steps
                + rwd_config.w_cosine_angle * math.cos(state.theta)
                + rwd_config.w_rel_distance
                * (1 - abs(state.x) / env_config.x_threshold)
            )
            rollout.append(RolloutEntry(state=state, action=action, reward=reward))
            if done:
                break

    return rollout
