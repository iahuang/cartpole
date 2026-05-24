"""Watch the trained REINFORCE policy play cartpole in realtime.

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
from models import CartPolePolicy

ARTIFACT_PATH = Path("./.artifacts/reinforce.pt")


def load_policy(pi: CartPolePolicy) -> bool:
    if not ARTIFACT_PATH.exists():
        return False
    pi.load_state_dict(torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=True))
    pi.eval()
    return True


def main() -> None:
    config = CartPoleConfig()
    env = CartPole(config)
    pi = CartPolePolicy()

    loaded = load_policy(pi)
    if not loaded:
        print(f"warning: no checkpoint at {ARTIFACT_PATH}, using random init")

    renderer = CartPoleRenderer(world_width=2 * config.x_threshold + 1.0)
    renderer.init("CartPole (REINFORCE) — R to reset+reload, Esc to quit")

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
            action = pi.sample(env.state, temperature=1.0)
        env.step(action)

        info = [
            f"x      = {env.state.x:+.2f} m",
            f"x_dot  = {env.state.x_dot:+.2f} m/s",
            f"theta  = {math.degrees(env.state.theta):+.1f}°",
            f"action = {action:+d}",
            f"steps  = {env.steps}",
            f"time   = {env.steps * config.dt:.2f} s",
        ]
        renderer.draw(env.state, config.pole_length, info=info)
        renderer.tick(fps)

    renderer.close()


if __name__ == "__main__":
    main()
