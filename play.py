"""Play cartpole with the keyboard.

Controls:
    A / Left arrow   — push cart left
    D / Right arrow  — push cart right
    R                — reset
    Esc / window-X   — quit
"""

import math

import pygame

from cartpole import CartPole, CartPoleConfig
from gfx import CartPoleRenderer


def main() -> None:
    config = CartPoleConfig()
    env = CartPole(config)
    renderer = CartPoleRenderer(world_width=2 * config.x_threshold + 1.0)
    renderer.init("CartPole — A/D or ←/→ to balance, R to reset")

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
                    env.reset()

        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action += 1

        if not env.done:
            env.step(action)

        info = [
            f"x      = {env.state.x:+.2f} m",
            f"x_dot  = {env.state.x_dot:+.2f} m/s",
            f"theta  = {math.degrees(env.state.theta):+.1f}°",
            f"steps  = {env.steps}",
            f"time   = {env.steps * config.dt:.2f} s",
        ]
        overlay = "GAME OVER — press R" if env.done else None
        renderer.draw(env.state, config.pole_length, info=info, overlay=overlay)
        renderer.tick(fps)

    renderer.close()


if __name__ == "__main__":
    main()
