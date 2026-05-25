"""Watch the trained PPO policy play cartpole in realtime, with a live
visualization of the policy network's activations and weighted contributions.

Controls:
    R                — reset (reloads latest checkpoint from disk)
    Esc / window-X   — quit
"""

import math
from pathlib import Path

import pygame
import torch

from cartpole import CartPole, CartPoleConfig
from gfx import CartPoleRenderer
from models import ACTION_DIM, STATE_FEATURE_DIM, CartPolePolicy, parameterize_state

ARTIFACT_PATH = Path("./.artifacts/ppo.pt")


def load_policy(pi: CartPolePolicy) -> bool:
    if not ARTIFACT_PATH.exists():
        return False
    pi.load_state_dict(torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=True))
    pi.eval()
    return True


class NetworkViz:
    """A tiny pygame-based renderer for the CartPolePolicy network.

    Shows three columns of neurons (input → hidden → output). Each neuron's
    color encodes the current activation (warm = positive, cool = negative,
    brightness ∝ |value| normalized per layer). Connection lines are colored
    by the sign of `w_ij * a_i` (the current per-connection contribution to
    the next layer), with alpha proportional to its magnitude.
    """

    INPUT_LABELS = ["x", "xd", "th", "thd"]
    OUTPUT_LABELS = ["L", "0", "R"]

    def __init__(self, pi: CartPolePolicy, panel_rect: pygame.Rect):
        self.pi = pi
        self.rect = panel_rect
        self.input_dim = STATE_FEATURE_DIM
        self.hidden_dim = pi.proj_up.out_features
        self.output_dim = ACTION_DIM

        self.font = pygame.font.SysFont(None, 18)
        self.label_font = pygame.font.SysFont(None, 16)

        col_xs = self._three_columns()
        self.input_pos = self._column_positions(col_xs[0], self.input_dim, margin_y=40)
        self.hidden_pos = self._column_positions(col_xs[1], self.hidden_dim, margin_y=20)
        self.output_pos = self._column_positions(col_xs[2], self.output_dim, margin_y=40)

    def _three_columns(self) -> tuple[int, int, int]:
        left = self.rect.left + 35
        right = self.rect.right - 35
        mid = (left + right) // 2
        return left, mid, right

    def _column_positions(self, x: int, n: int, *, margin_y: int) -> list[tuple[int, int]]:
        top = self.rect.top + margin_y
        bottom = self.rect.bottom - margin_y
        if n == 1:
            return [(x, (top + bottom) // 2)]
        return [
            (x, int(top + i * (bottom - top) / (n - 1)))
            for i in range(n)
        ]

    @staticmethod
    def _activation_color(value: float, scale: float) -> tuple[int, int, int]:
        # Diverging map: positive → warm orange, negative → cool blue, zero → grey.
        v = max(-1.0, min(1.0, value / (scale + 1e-6)))
        if v >= 0:
            r = int(80 + 175 * v)
            g = int(80 + 70 * v)
            b = int(80 - 30 * v)
        else:
            t = -v
            r = int(80 - 30 * t)
            g = int(80 + 70 * t)
            b = int(80 + 175 * t)
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    def _draw_connections(
        self,
        surf: pygame.Surface,
        src_pos: list[tuple[int, int]],
        dst_pos: list[tuple[int, int]],
        weight: torch.Tensor,  # (n_dst, n_src)
        src_activation: torch.Tensor,  # (n_src,)
    ) -> None:
        # Per-connection contribution to the dst neuron: w_ij * a_i.
        contrib = weight * src_activation.unsqueeze(0)  # (n_dst, n_src)
        max_c = contrib.abs().max().item() + 1e-6

        for j in range(weight.shape[0]):
            for i in range(weight.shape[1]):
                c = contrib[j, i].item()
                strength = abs(c) / max_c
                if strength < 0.04:
                    continue
                alpha = int(255 * strength)
                if c >= 0:
                    color = (220, 90, 60, alpha)   # warm
                else:
                    color = (60, 130, 220, alpha)  # cool
                pygame.draw.line(surf, color, src_pos[i], dst_pos[j], 1)

    def _draw_neurons(
        self,
        screen: pygame.Surface,
        positions: list[tuple[int, int]],
        activations: torch.Tensor,
        *,
        radius: int,
        labels: list[str] | None = None,
        label_side: str = "left",
        highlight: int | None = None,
    ) -> None:
        scale = activations.abs().max().item() + 1e-6
        for i, pos in enumerate(positions):
            v = activations[i].item()
            color = self._activation_color(v, scale)
            r = radius + (3 if highlight == i else 0)
            pygame.draw.circle(screen, color, pos, r)
            outline = (20, 20, 20) if highlight == i else (60, 60, 70)
            pygame.draw.circle(screen, outline, pos, r, 2 if highlight == i else 1)
            if labels is not None:
                label = self.label_font.render(labels[i], True, (40, 40, 50))
                if label_side == "left":
                    screen.blit(label, (pos[0] - r - 4 - label.get_width(), pos[1] - 7))
                else:
                    screen.blit(label, (pos[0] + r + 4, pos[1] - 7))

    def draw(self, screen: pygame.Surface, state) -> None:
        # Forward pass with intermediate activations.
        with torch.no_grad():
            inp = parameterize_state(state)
            h_pre = self.pi.proj_up(inp)
            hid = torch.relu(h_pre)
            logits = self.pi.proj_down(hid)

        # Panel background.
        bg_rect = self.rect.inflate(0, 0)
        pygame.draw.rect(screen, (252, 252, 255), bg_rect, border_radius=6)
        pygame.draw.rect(screen, (180, 180, 195), bg_rect, width=1, border_radius=6)

        title = self.font.render("policy network", True, (60, 60, 80))
        screen.blit(title, (self.rect.left + 10, self.rect.top + 6))

        # Connections drawn on an alpha-enabled surface so we get translucent lines.
        line_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        self._draw_connections(
            line_surf,
            self.input_pos,
            self.hidden_pos,
            self.pi.proj_up.weight.detach(),
            inp,
        )
        self._draw_connections(
            line_surf,
            self.hidden_pos,
            self.output_pos,
            self.pi.proj_down.weight.detach(),
            hid,
        )
        screen.blit(line_surf, (0, 0))

        # Neurons.
        self._draw_neurons(
            screen, self.input_pos, inp,
            radius=9, labels=self.INPUT_LABELS, label_side="left",
        )
        self._draw_neurons(
            screen, self.hidden_pos, hid,
            radius=6,
        )
        argmax = int(logits.argmax().item())
        self._draw_neurons(
            screen, self.output_pos, logits,
            radius=10, labels=self.OUTPUT_LABELS, label_side="right",
            highlight=argmax,
        )


def main() -> None:
    config = CartPoleConfig()
    env = CartPole(config)
    pi = CartPolePolicy()

    loaded = load_policy(pi)
    if not loaded:
        print(f"warning: no checkpoint at {ARTIFACT_PATH}, using random init")

    window_w, window_h = 1100, 420
    renderer = CartPoleRenderer(
        width=window_w,
        height=window_h,
        world_width=2 * config.x_threshold + 1.0,
        cart_center_x=360,  # leaves the right side free for the network viz
    )
    renderer.init("CartPole (PPO) — R to reset+reload, Esc to quit")

    viz_panel = pygame.Rect(740, 20, 340, 380)
    viz = NetworkViz(pi, viz_panel)

    fps = round(1.0 / config.dt)
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    load_policy(pi)
                    env.reset()

        if env.done:
            load_policy(pi)
            env.reset()

        with torch.no_grad():
            action = pi.sample(env.state, temperature=1.0)[0]
        env.step(action)

        info = [
            f"x      = {env.state.x:+.2f} m",
            f"x_dot  = {env.state.x_dot:+.2f} m/s",
            f"theta  = {math.degrees(env.state.theta):+.1f}°",
            f"action = {action:+d}",
            f"steps  = {env.steps}",
            f"time   = {env.steps * config.dt:.2f} s",
        ]
        renderer.draw(
            env.state,
            config.pole_length,
            info=info,
            extra_draw=lambda s: viz.draw(s, env.state),
        )
        renderer.tick(fps)

    renderer.close()


if __name__ == "__main__":
    main()
