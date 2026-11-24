"""Train DDPG on the TruckMDPEnv (15-dim state, steering in [-1, 1]).

Run from the repo root:
    python -m train.train_ddpg --course straight --velocity -2.012 --episodes 500
"""
import argparse
import os

import gymnasium as gym
import numpy as np
import torch

import gym_truck_backerupper  # noqa: F401  (registers TruckBackerUpper-v0)
from gym_truck_backerupper.mdp.wrap_env import GymBackerUpperBackend, TruckMDPEnv
from gym_truck_backerupper.metrics import TrainingEpisodeRecord, TrainingMetricsDirectory

from .ddpg import DDPGAgent
from .replay_buffer import ReplayBuffer

# ((x0, y0, heading_deg), (xg, yg, heading_deg)) -- same convention as manual_control_pygame.py
COURSES = {
    "straight": ((28.0, 0.0, 180.0), (-28.0, 0.0, 180.0)),
    "turn": ((28.0, -18.0, 180.0), (-8.0, 22.0, 90.0)),
    "scurve": ((30.0, -16.0, 180.0), (-30.0, 16.0, 180.0)),
    "random": None,  # simulator draws random start/goal (generalisation test)
}
LAT_SCALE = 5.0  # same as ObsConfig.lat_scale, used to un-normalize obs[2]


def make_env(course_name, velocity) -> TruckMDPEnv:
    raw = gym.make("TruckBackerUpper-v0")
    backend = GymBackerUpperBackend(raw, velocity=velocity, course=COURSES[course_name])
    return TruckMDPEnv(backend)




def run_episode(env: TruckMDPEnv, agent: DDPGAgent, buffer: ReplayBuffer, sigma, total_steps, warmup, batch_size, first_seed=None, train=True):
    obs, _ = env.reset(seed=first_seed)
    ep_reward, losses, lat_errs, steer = 0.0, [], [], [obs[4]]
    info, done = {}, False
    while not done:
        if train and total_steps < warmup:
            action = env.action_space.sample()          # pure exploration at the start
        else:
            action = agent.act(obs, noise_sigma=sigma if train else 0.0)
        obs2, r, terminated, truncated, info = env.step(action)
        if train:
            buffer.add(obs, action, r, obs2, terminated)
            if total_steps >= warmup and len(buffer) >= batch_size:
                losses.append(agent.update(buffer.sample(batch_size))[0])
            total_steps += 1
        ep_reward += r
        lat_errs.append(abs(obs2[2]) * LAT_SCALE)
        steer.append(obs2[4])
        obs, done = obs2, bool(terminated or truncated)
    precision = float(np.mean(lat_errs))                     # mean |lateral error| in metres
    stability = float(np.mean(np.abs(np.diff(steer))))       # mean steering variation (zig-zag)
    return ep_reward, (float(np.mean(losses)) if losses else float("nan")), \
        bool(info.get("success", False)), precision, stability, total_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--course", choices=COURSES, default="straight")
    p.add_argument("--velocity", type=float, default=-2.012, help="<0 reverse, >0 forward")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sigma-start", type=float, default=0.3)
    p.add_argument("--sigma-end", type=float, default=0.05)
    p.add_argument("--sigma-decay-steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="ddpg_straight_reverse")
    p.add_argument("--init-actor", default=None, help="BC actor weights (.pt) for Imitation init")
    p.add_argument("--demos", default=None, help="npz with s,a,r,s2,done (see README)")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, eval_env = make_env(args.course, args.velocity), make_env(args.course, args.velocity)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    agent = DDPGAgent(obs_dim, act_dim)
    buffer = ReplayBuffer(obs_dim, act_dim)
    if args.init_actor:
        agent.load_actor(args.init_actor)
    if args.demos:
        d = np.load(args.demos)
        buffer.add_demonstrations(d["s"], d["a"], d["r"], d["s2"], d["done"])
        print(f"{buffer.n_demo} demonstration transitions loaded")

    metrics = TrainingMetricsDirectory("./training_metrics").get_document(args.run_name)
    out_dir = os.path.join("runs", args.run_name); os.makedirs(out_dir, exist_ok=True)

    total_steps, best_sr = 0, -1.0
    for ep in range(1, args.episodes + 1):
        frac = min(1.0, total_steps / args.sigma_decay_steps)
        sigma = args.sigma_start + frac * (args.sigma_end - args.sigma_start)

        R, loss, ok, prec, stab, total_steps = run_episode(env, agent, buffer, sigma, total_steps, 
                                                           args.warmup, args.batch_size,
                                                           first_seed=args.seed if ep == 1 else None)

        metrics.add_record(TrainingEpisodeRecord(ep, R, loss, ok, prec, stab))
        print(f"ep {ep:4d} | steps {total_steps:7d} | R {R:9.1f} | loss {loss:8.3f} | "
              f"success {ok} | |e_lat| {prec:5.2f} m | sigma {sigma:.2f}")

        if ep % args.eval_every == 0:
            wins = [run_episode(eval_env, agent, buffer, 0.0, 0, 0, 0, train=False)[2] for _ in range(10)]
            sr = float(np.mean(wins))
            print(f"   >>> eval success rate: {sr:.0%}")
            agent.save(os.path.join(out_dir, "last.pt"))
            if sr >= best_sr:
                best_sr = sr; agent.save(os.path.join(out_dir, "best.pt"))
    env.close(); eval_env.close()


if __name__ == "__main__":
    main()
