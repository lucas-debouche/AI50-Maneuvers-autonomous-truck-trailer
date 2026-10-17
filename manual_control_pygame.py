"""Manual Pygame control for TruckBackerUpper-v0.

Controls:
  Left/Right or A/D : steer
  1 : straight line
  2 : constant turn
  3 : S-curve
  R : reset current track
  F : forward / reverse
  Esc/Q : quit
"""
import math
import sys

import gymnasium as gym
import gym_truck_backerupper  # noqa: F401
import numpy as np
import pygame


W, H = 900, 900
SCALE = 11.0
FPS = 60
PHYS_STEPS_PER_FRAME = 1
STEER_RATE = np.radians(35.0)
DEFAULT_SPEED = 0.45
MAX_STEER_DEG = 25.0

# q0 / qg = [x, y, heading_deg]
COURSES = [
    ("1 Straight", [28.0, 0.0, 180.0], [-28.0, 0.0, 180.0]),
    ("2 Constant turn", [28.0, -18.0, 180.0], [-8.0, 22.0, 90.0]),
    ("3 S-curve", [30.0, -16.0, 180.0], [-30.0, 16.0, 180.0]),
]


def world_to_screen(x, y, cam_x, cam_y):
    sx = int(W / 2 + (x - cam_x) * SCALE)
    sy = int(H / 2 - (y - cam_y) * SCALE)
    return sx, sy


def draw_oriented_rect(surface, x, y, length, width, angle, color, cam_x, cam_y):
    c, s = math.cos(angle), math.sin(angle)
    hx, hy = length * 0.5, width * 0.5
    local = [(-hx * 0.3, -hy), (hx * 1.2, -hy), (hx * 1.2, hy), (-hx * 0.3, hy)]
    pts = []
    for lx, ly in local:
        wx = x + lx * c - ly * s
        wy = y + lx * s + ly * c
        pts.append(world_to_screen(wx, wy, cam_x, cam_y))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (20, 20, 20), pts, 2)


def draw_path(surface, track, cam_x, cam_y):
    if track is None or len(track) < 2:
        return
    step = max(1, len(track) // 500)
    pts = [world_to_screen(p[0], p[1], cam_x, cam_y) for p in track[::step]]
    if len(pts) > 1:
        pygame.draw.lines(surface, (80, 160, 255), False, pts, 3)


def draw_dock(surface, env, cam_x, cam_y):
    if not hasattr(env, "dock_x"):
        return
    pts = [world_to_screen(x, y, cam_x, cam_y) for x, y in zip(env.dock_x, env.dock_y)]
    if len(pts) > 1:
        pygame.draw.lines(surface, (220, 60, 60), False, pts, 4)


def apply_easy_params(raw, forward: bool):
    raw.manual_params(L2=7.0, h=0.0)
    raw.H_c = raw.L2 / 3.0
    raw.H_t = raw.L2 / 3.0
    raw.max_hitch = np.radians(110.0)
    raw.max_path_error = 12.0
    raw.max_heading_error = np.radians(70.0)
    raw.manual_velocity(DEFAULT_SPEED if forward else -DEFAULT_SPEED)
    raw.manual_offset(0.0, 0.0, 0.0)


def load_course(raw, course_idx: int, forward: bool):
    name, q0, qg = COURSES[course_idx]
    apply_easy_params(raw, forward)
    raw.manual_course(q0, qg)
    return name


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Truck Backer-Upper — manual control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    env = gym.make("TruckBackerUpper-v0")
    raw = env.unwrapped

    course_idx = 0
    forward = True
    course_name = load_course(raw, course_idx, forward)
    obs, info = env.reset()
    steer = 0.0
    total_reward = 0.0
    running = True
    done = False
    max_steer = np.radians(MAX_STEER_DEG)

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    course_name = load_course(raw, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                elif event.key == pygame.K_f:
                    forward = not forward
                    course_name = load_course(raw, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    course_idx = 0
                    course_name = load_course(raw, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    course_idx = 1
                    course_name = load_course(raw, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    course_idx = 2
                    course_name = load_course(raw, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= STEER_RATE * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += STEER_RATE * dt
        if not (keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_RIGHT] or keys[pygame.K_d]):
            steer *= max(0.0, 1.0 - 3.0 * dt)
        steer = float(np.clip(steer, -max_steer, max_steer))

        if not done:
            for _ in range(PHYS_STEPS_PER_FRAME):
                obs, reward, terminated, truncated, info = env.step(
                    np.array([steer], dtype=np.float32)
                )
                total_reward += float(reward)
                done = bool(terminated or truncated)
                if done:
                    break
        else:
            info = info if isinstance(info, dict) else {}

        i = max(raw.sim_i - 1, 0)
        x1, y1, psi1 = float(raw.x1[i]), float(raw.y1[i]), float(raw.psi_1[i])
        x2, y2, psi2 = float(raw.x2[i]), float(raw.y2[i]), float(raw.psi_2[i])
        cam_x, cam_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        screen.fill((32, 36, 40))
        for g in range(-40, 41, 5):
            pygame.draw.line(
                screen, (45, 50, 55),
                world_to_screen(g, -40, cam_x, cam_y),
                world_to_screen(g, 40, cam_x, cam_y), 1,
            )
            pygame.draw.line(
                screen, (45, 50, 55),
                world_to_screen(-40, g, cam_x, cam_y),
                world_to_screen(40, g, cam_x, cam_y), 1,
            )

        draw_path(screen, getattr(raw, "track_vector", None), cam_x, cam_y)
        draw_dock(screen, raw, cam_x, cam_y)
        draw_oriented_rect(screen, x2, y2, raw.L2, raw.H_t, psi2, (200, 160, 40), cam_x, cam_y)
        draw_oriented_rect(screen, x1, y1, raw.L1, raw.H_c, psi1, (60, 170, 90), cam_x, cam_y)
        pygame.draw.line(
            screen,
            (240, 240, 240),
            world_to_screen(x1, y1, cam_x, cam_y),
            world_to_screen(x2, y2, cam_x, cam_y),
            2,
        )

        hitch = math.degrees(((psi1 - psi2 + math.pi) % (2 * math.pi)) - math.pi)
        dir_txt = "FORWARD" if forward else "REVERSE"
        lines = [
            f"{course_name} | {dir_txt} | reward {total_reward:7.1f}",
            f"steer {math.degrees(steer):+5.1f} deg | hitch {hitch:+5.1f} deg | done={done}",
            "1/2/3 tracks | arrows steer | R reset | F fwd/rev | Q quit",
        ]
        if done and info:
            flags = [k for k, v in info.items() if v is True]
            if flags:
                lines.append("end: " + ", ".join(flags))

        y_txt = 12
        for line in lines:
            screen.blit(font.render(line, True, (230, 230, 230)), (12, y_txt))
            y_txt += 22

        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
