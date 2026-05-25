"""Small Pygame-based renderer for the cartpole environment.

Decoupled from the sim: pass in a CartPoleState (or anything with x/theta
attributes) plus the pole half-length, and it draws a frame.
"""

import math
from typing import Callable

import pygame


class CartPoleRenderer:
    def __init__(
        self,
        width: int = 800,
        height: int = 420,
        world_width: float = 5.0,
        track_y_frac: float = 0.7,
        cart_center_x: int | None = None,
    ):
        self.width = width
        self.height = height
        self.world_width = world_width
        self.scale = width / world_width  # pixels per meter
        self.track_y = int(height * track_y_frac)
        self.cart_center_x = cart_center_x if cart_center_x is not None else width // 2

        self.bg = (245, 246, 250)
        self.track_color = (60, 60, 70)
        self.cart_color = (50, 90, 160)
        self.pole_color = (190, 95, 50)
        self.pivot_color = (30, 30, 30)
        self.text_color = (30, 30, 30)
        self.overlay_color = (200, 40, 40)

        self.cart_w = 70
        self.cart_h = 38
        self.pole_thickness = 8

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.big_font: pygame.font.Font | None = None

    def init(self, caption: str = "CartPole") -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.big_font = pygame.font.SysFont(None, 44)

    def _world_to_screen_x(self, x: float) -> int:
        return int(self.cart_center_x + x * self.scale)

    def draw(
        self,
        state,
        pole_length: float,
        info: list[str] | None = None,
        overlay: str | None = None,
        *,
        extra_draw: Callable[[pygame.Surface], None] | None = None,
    ) -> None:
        assert self.screen is not None, "call init() before draw()"
        self.screen.fill(self.bg)

        pygame.draw.line(
            self.screen,
            self.track_color,
            (0, self.track_y),
            (self.width, self.track_y),
            2,
        )

        cart_cx = self._world_to_screen_x(state.x)
        cart_cy = self.track_y - self.cart_h // 2
        cart_rect = pygame.Rect(0, 0, self.cart_w, self.cart_h)
        cart_rect.center = (cart_cx, cart_cy)
        pygame.draw.rect(self.screen, self.cart_color, cart_rect, border_radius=5)

        pivot = (cart_cx, cart_cy)
        pole_pix = 2.0 * pole_length * self.scale
        end_x = pivot[0] + pole_pix * math.sin(state.theta)
        end_y = pivot[1] - pole_pix * math.cos(state.theta)
        pygame.draw.line(
            self.screen,
            self.pole_color,
            pivot,
            (end_x, end_y),
            self.pole_thickness,
        )
        pygame.draw.circle(self.screen, self.pivot_color, pivot, 5)

        if info:
            y = 10
            for line in info:
                surf = self.font.render(line, True, self.text_color)
                self.screen.blit(surf, (12, y))
                y += 22

        if overlay:
            surf = self.big_font.render(overlay, True, self.overlay_color)
            rect = surf.get_rect(center=(self.width // 2, self.height // 2 - 40))
            self.screen.blit(surf, rect)

        if extra_draw is not None:
            extra_draw(self.screen)

        pygame.display.flip()

    def tick(self, fps: float) -> None:
        assert self.clock is not None, "call init() before tick()"
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
