"""Manual Pygame control for TruckBackerUpper via the MDP layer.

Records demonstrations compatible with TruckMDP (15-D obs, action in [-1, 1]).

Controls:
  Left/Right or A/D : steer
  1 : straight line
  2 : circular trajectory
  3 : S-shaped trajectory
  R : reset current track
  F : forward / reverse
  Esc/Q : quit
  E : start recording a demonstration
  V : validate and save (only if success / goal reached)
  X : abandon the demonstration
"""
import csv
import math
import os
import sys

import gymnasium as gym
import gym_truck_backerupper  # noqa: F401
import numpy as np
import pygame

from gym_truck_backerupper.mdp.wrap_env import GymBackerUpperBackend, TruckMDPEnv


W, H = 900, 900
SCALE = 11.0
FPS = 60
PHYS_STEPS_PER_FRAME = 1
STEER_RATE = np.radians(35.0)
DEFAULT_SPEED = 0.35  # a bit slower = easier trailer control
MAX_STEER_DEG = 25.0
DEMO_FILE = "demonstrations.csv"

# Names of the 15 MDP observation dimensions (same order as TruckMDP.observation)
OBS_NAMES = [
    "tractor_heading_err",
    "trailer_heading_err",
    "trailer_lat_err",
    "hitch",
    "steering",
    "velocity",
    "preview_x1",
    "preview_y1",
    "preview_x2",
    "preview_y2",
    "preview_x3",
    "preview_y3",
    "goal_x",
    "goal_y",
    "goal_heading",
]

# q0 / qg = [x, y, heading_deg]
# Matches the project trajectories:
#   1) straight line
#   2) circular (~90° arc, shortest valid Dubins-like setup found)
#   3) S-shaped (opposite lateral offsets, same final heading)
COURSES = [
    ("1 Straight line", [28.0, 0.0, 180.0], [-28.0, 0.0, 180.0]),
    ("2 Circular", [28.0, -20.0, 180.0], [2.0, 22.0, 90.0]),
    ("3 S-shaped", [28.0, -8.0, 180.0], [-28.0, 8.0, 180.0]),
]


def world_to_screen(x, y, cam_x, cam_y):
    sx = int(W / 2 + (x - cam_x) * SCALE)
    sy = int(H / 2 - (y - cam_y) * SCALE)
    return sx, sy


def draw_body(surface, x, y, rear, front, width, angle, color, cam_x, cam_y):
    """Draw a vehicle body in the heading frame: rear < 0 < front along x."""
    c, s = math.cos(angle), math.sin(angle)
    hy = width * 0.5
    local = [(rear, -hy), (front, -hy), (front, hy), (rear, hy)]
    pts = []
    for lx, ly in local:
        wx = x + lx * c - ly * s
        wy = y + lx * s + ly * c
        pts.append(world_to_screen(wx, wy, cam_x, cam_y))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (20, 20, 20), pts, 2)


HITCH_GAP = 0.45  # visual gap between tractor rear and trailer front


def draw_hitch_link(
    surface, x1, y1, psi1, x2, y2, psi2, L2, cam_x, cam_y
):
    """Draw the hitch joint in the gap between trailer front and tractor rear."""
    c2, s2 = math.cos(psi2), math.sin(psi2)
    c1, s1 = math.cos(psi1), math.sin(psi1)
    # Hitch is at the trailer front attachment (equals tractor ref when h=0)
    hx = x2 + L2 * c2
    hy = y2 + L2 * s2
    trailer_front = (x2 + (L2 - HITCH_GAP) * c2, y2 + (L2 - HITCH_GAP) * s2)
    tractor_rear = (x1 + HITCH_GAP * c1, y1 + HITCH_GAP * s1)
    pts = [
        world_to_screen(*trailer_front, cam_x, cam_y),
        world_to_screen(hx, hy, cam_x, cam_y),
        world_to_screen(*tractor_rear, cam_x, cam_y),
    ]
    pygame.draw.lines(surface, (230, 230, 230), False, pts, 3)
    pygame.draw.circle(surface, (230, 230, 230), world_to_screen(hx, hy, cam_x, cam_y), 4)


def draw_path(surface, track, cam_x, cam_y):
    if track is None or len(track) < 2:
        return
    step = max(1, len(track) // 500)
    pts = [world_to_screen(p[0], p[1], cam_x, cam_y) for p in track[::step]]
    if len(pts) > 1:
        pygame.draw.lines(surface, (80, 160, 255), False, pts, 3)


def draw_dock(surface, sim, cam_x, cam_y):
    if not hasattr(sim, "dock_x"):
        return
    pts = [world_to_screen(x, y, cam_x, cam_y) for x, y in zip(sim.dock_x, sim.dock_y)]
    if len(pts) > 1:
        pygame.draw.lines(surface, (220, 60, 60), False, pts, 4)


def apply_easy_params(sim, forward: bool):
    sim.manual_params(L2=7.0, h=0.0)
    sim.H_c = sim.L2 / 3.0
    sim.H_t = sim.L2 / 3.0
    sim.max_hitch = np.radians(110.0)
    sim.max_path_error = 12.0
    sim.max_heading_error = np.radians(70.0)
    sim.manual_velocity(DEFAULT_SPEED if forward else -DEFAULT_SPEED)
    sim.manual_offset(0.0, 0.0, 0.0)


def load_course(backend, course_idx: int, forward: bool):
    name, q0, qg = COURSES[course_idx]
    apply_easy_params(backend.sim, forward)
    # Backend re-applies q0/qg on every reset (avoids reverse mutating the arrays)
    backend.course = (list(q0), list(qg))
    return name


def obs_to_dict(obs, prefix: str) -> dict:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[0] != len(OBS_NAMES):
        raise ValueError(
            f"Expected {len(OBS_NAMES)}-D MDP obs, got shape {obs.shape}"
        )
    return {f"{prefix}_{name}": float(obs[i]) for i, name in enumerate(OBS_NAMES)}


def next_episode_id(filename: str = DEMO_FILE) -> int:
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        return 1
    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        ids = [int(row["episode_id"]) for row in reader if row.get("episode_id")]
    return (max(ids) + 1) if ids else 1


def save_demonstrations(rows, episode_id: int):
    if not rows:
        return

    for row in rows:
        row["episode_id"] = episode_id

    file_exists = os.path.isfile(DEMO_FILE) and os.path.getsize(DEMO_FILE) > 0
    with open(DEMO_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Episode {episode_id}: {len(rows)} samples saved to {DEMO_FILE}")


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Truck Backer-Upper — MDP recording")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    base_env = gym.make("TruckBackerUpper-v0")
    backend = GymBackerUpperBackend(base_env)
    env = TruckMDPEnv(backend)
    sim = backend.sim
    max_steer = float(backend.max_steer)  # rad (45°)
    human_max_steer = np.radians(MAX_STEER_DEG)

    course_idx = 0
    forward = True
    course_name = load_course(backend, course_idx, forward)
    obs, info = env.reset()
    steer = 0.0
    total_reward = 0.0
    running = True
    done = False
    recording = False
    recording_rows = []
    reached_goal = False
    episode_id = next_episode_id()

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_e:
                    if not recording:
                        recording = True
                        recording_rows = []
                        reached_goal = False
                        print(f"Demonstration started (episode_id={episode_id}).")
                        print("Reach the goal, then V = validate | X = discard")
                    else:
                        print("Finish with V or discard with X.")

                elif event.key == pygame.K_v:
                    if not recording:
                        print("No demonstration in progress.")
                    elif not reached_goal:
                        print(
                            "Cannot validate: goal not reached. "
                            "Keep driving or press X to discard."
                        )
                    else:
                        save_demonstrations(recording_rows, episode_id)
                        episode_id += 1
                        recording = False
                        recording_rows = []
                        reached_goal = False
                        print("Demonstration VALIDATED.")

                elif event.key == pygame.K_x:
                    if recording:
                        print(
                            f"Demonstration DISCARDED: "
                            f"{len(recording_rows)} samples deleted."
                        )
                        recording = False
                        recording_rows = []
                        reached_goal = False
                    else:
                        print("No demonstration in progress.")

                elif recording:
                    # Avoid mixing courses / direction inside one demo
                    print("Recording in progress: finish with V or discard with X.")

                elif event.key == pygame.K_r:
                    course_name = load_course(backend, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                    reached_goal = False

                elif event.key == pygame.K_f:
                    forward = not forward
                    course_name = load_course(backend, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                    reached_goal = False

                elif event.key in (pygame.K_1, pygame.K_KP1):
                    course_idx = 0
                    course_name = load_course(backend, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                    reached_goal = False

                elif event.key in (pygame.K_2, pygame.K_KP2):
                    course_idx = 1
                    course_name = load_course(backend, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                    reached_goal = False

                elif event.key in (pygame.K_3, pygame.K_KP3):
                    course_idx = 2
                    course_name = load_course(backend, course_idx, forward)
                    obs, info = env.reset()
                    steer = 0.0
                    total_reward = 0.0
                    done = False
                    reached_goal = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= STEER_RATE * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += STEER_RATE * dt
        if not (
            keys[pygame.K_LEFT]
            or keys[pygame.K_a]
            or keys[pygame.K_RIGHT]
            or keys[pygame.K_d]
        ):
            steer *= max(0.0, 1.0 - 3.0 * dt)
        steer = float(np.clip(steer, -human_max_steer, human_max_steer))

        if not done:
            for _ in range(PHYS_STEPS_PER_FRAME):
                obs_before = np.asarray(obs, dtype=np.float32).reshape(-1).copy()

                # MDP expects a normalized action in [-1, 1]
                action_norm = float(np.clip(steer / max_steer, -1.0, 1.0))
                action = np.array([action_norm], dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                done = bool(terminated or truncated)
                if info.get("success", False):
                    reached_goal = True

                if recording:
                    row = {
                        "episode_id": episode_id,
                        "course": course_name,
                        "direction": "forward" if forward else "reverse",
                        **obs_to_dict(obs_before, "obs"),
                        "action_steer": action_norm,
                        "reward": float(reward),
                        **obs_to_dict(obs, "next_obs"),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                    }
                    recording_rows.append(row)

                if done:
                    break
        else:
            info = info if isinstance(info, dict) else {}

        i = max(sim.sim_i - 1, 0)
        x1, y1, psi1 = float(sim.x1[i]), float(sim.y1[i]), float(sim.psi_1[i])
        x2, y2, psi2 = float(sim.x2[i]), float(sim.y2[i]), float(sim.psi_2[i])
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

        draw_path(screen, getattr(sim, "track_vector", None), cam_x, cam_y)
        draw_dock(screen, sim, cam_x, cam_y)
        # Physics: hitch ~ tractor axle (h=0). Leave a visible gap around the hitch.
        trail = float(getattr(sim, "trail", 2.0))
        draw_body(
            screen, x2, y2,
            -trail, float(sim.L2) - HITCH_GAP,
            sim.H_t, psi2, (200, 160, 40), cam_x, cam_y,
        )
        draw_body(
            screen, x1, y1,
            HITCH_GAP, float(sim.L1),
            sim.H_c, psi1, (60, 170, 90), cam_x, cam_y,
        )
        draw_hitch_link(
            screen, x1, y1, psi1, x2, y2, psi2, float(sim.L2), cam_x, cam_y
        )

        hitch = math.degrees(((psi1 - psi2 + math.pi) % (2 * math.pi)) - math.pi)
        dir_txt = "FORWARD" if forward else "REVERSE"
        lines = [
            f"{course_name} | {dir_txt} | reward {total_reward:7.1f} | MDP 15D",
            f"steer {math.degrees(steer):+5.1f} deg | hitch {hitch:+5.1f} deg | done={done}",
            "1/2/3 tracks | arrows steer | R reset | F fwd/rev | Q quit",
            "E start recording | V validate (goal only) | X abandon",
        ]
        if recording:
            lines.insert(
                0,
                f"REC episode {episode_id} | samples {len(recording_rows)} | "
                f"goal={'YES' if reached_goal else 'no'}",
            )
        if done and info:
            flags = [k for k, v in info.items() if v is True]
            if flags:
                lines.append("end: " + ", ".join(flags))

        y_txt = 12
        for line in lines:
            color = (255, 80, 80) if line.startswith("REC") else (230, 230, 230)
            screen.blit(font.render(line, True, color), (12, y_txt))
            y_txt += 22

        pygame.display.flip()

    if recording and recording_rows:
        print("Demonstration not validated.")

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
