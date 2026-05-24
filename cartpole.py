import math
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CartPoleConfig:
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5  # half-length: pivot to pole center of mass
    force_mag: float = 8.0
    dt: float = 0.02  # time step (default 50 Hz)
    x_threshold: float = 2.4
    theta_threshold: float = math.radians(50.0)
    init_perturb: float = (
        0.05  # max |value| for uniform init noise on each state component
    )


@dataclass
class CartPoleState:
    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0


class CartPole:
    """Headless cartpole simulator using semi-implicit Euler integration.

    State convention: theta = 0 is upright; positive theta tilts to +x side.
    Action is a scalar in [-1, 1] (clamped); force applied to cart = action * force_mag.
    """

    def __init__(
        self, config: CartPoleConfig | None = None, *, seed: int | None = None
    ):
        self.config = config or CartPoleConfig()
        self._rng = random.Random(seed)
        self.state = CartPoleState()
        self.steps = 0
        self.done = False
        self.reset()

    def reset(self, state: CartPoleState | None = None) -> CartPoleState:
        if state is not None:
            self.state = replace(state)
        else:
            p = self.config.init_perturb
            self.state = CartPoleState(
                x=self._rng.uniform(-p, p),
                x_dot=self._rng.uniform(-p, p),
                theta=self._rng.uniform(-p, p),
                theta_dot=self._rng.uniform(-p, p),
            )
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action: float) -> tuple[CartPoleState, bool]:
        c = self.config
        action = max(-1.0, min(1.0, float(action)))
        force = action * c.force_mag

        x, x_dot, theta, theta_dot = (
            self.state.x,
            self.state.x_dot,
            self.state.theta,
            self.state.theta_dot,
        )

        total_mass = c.cart_mass + c.pole_mass
        polemass_length = c.pole_mass * c.pole_length
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        temp = (force + polemass_length * theta_dot * theta_dot * sin_t) / total_mass
        theta_acc = (c.gravity * sin_t - cos_t * temp) / (
            c.pole_length * (4.0 / 3.0 - c.pole_mass * cos_t * cos_t / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_t / total_mass

        x_dot += c.dt * x_acc
        x += c.dt * x_dot
        theta_dot += c.dt * theta_acc
        theta += c.dt * theta_dot

        self.state = CartPoleState(x, x_dot, theta, theta_dot)
        self.steps += 1
        self.done = abs(x) > c.x_threshold or abs(theta) > c.theta_threshold
        return self.state, self.done
