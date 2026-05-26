import math
from typing import override

import torch

from cartpole import CartPoleState


def parameterize_state(state: CartPoleState) -> torch.Tensor:
    # θ is encoded as (sin θ, cos θ) so the network sees a bounded,
    # continuous, 2π-periodic angle representation — important for swing-up
    # where the pole can wrap around through ±π.
    return torch.tensor(
        [
            state.x,
            state.x_dot,
            math.sin(state.theta),
            math.cos(state.theta),
            state.theta_dot,
        ],
        dtype=torch.float32,
    )


STATE_FEATURE_DIM = 5
ACTION_DIM = 3


class CartPolePolicy(torch.nn.Module):
    proj_up: torch.nn.Linear
    proj_down: torch.nn.Linear

    def __init__(self, *, hidden_dim: int = 16):
        super().__init__()
        self.proj_up = torch.nn.Linear(
            STATE_FEATURE_DIM,
            hidden_dim,
        )
        self.proj_down = torch.nn.Linear(
            hidden_dim,
            3,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_up(x)
        x = torch.relu(x)
        x = self.proj_down(x)
        return x

    @override
    def sample(
        self, state: CartPoleState, *, temperature: float = 1.0
    ) -> tuple[int, int, torch.Tensor]:
        logits = self(parameterize_state(state))
        probs = torch.softmax(logits / temperature, dim=0)

        class_idx = int(torch.multinomial(probs, num_samples=1).item())

        return class_idx - 1, class_idx, logits  # map class index to action (-1, 0, 1)


class CartPoleValue(torch.nn.Module):
    proj_up: torch.nn.Linear
    proj_down: torch.nn.Linear

    def __init__(self, *, hidden_dim: int = 16):
        super().__init__()
        self.proj_up = torch.nn.Linear(
            STATE_FEATURE_DIM,
            hidden_dim,
        )
        self.proj_down = torch.nn.Linear(
            hidden_dim,
            1,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_up(x)
        x = torch.relu(x)
        x = self.proj_down(x)
        return x.squeeze(-1)
