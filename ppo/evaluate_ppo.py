
from pathlib import Path

import gymnasium as gym
import gym_truck_backerupper
import numpy as np

from stable_baselines3 import PPO

from gym_truck_backerupper.mdp.wrap_env import (
    GymBackerUpperBackend,
    TruckMDPEnv,
)


def main():
    # 1. Load the previously trained PPO model
    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "ppo_reverse_50k_terminalfix.zip"
    )

    model = PPO.load(str(model_path), device="cpu")

    # 2. Create the evaluation environment
    base_env = gym.make("TruckBackerUpper-v0")
    backend = GymBackerUpperBackend(base_env, velocity=-2.012)
    env = TruckMDPEnv(backend)

    n_episodes = 10
    episode_rewards = []
    successes = 0

    try:
        # 3. Evaluate the trained agent
        for episode in range(n_episodes):
            obs, info = env.reset(seed=100 + episode)

            total_reward = 0.0
            steps = 0

            for _ in range(2500):
                # Deterministic action: no exploration noise
                action, _ = model.predict(
                    obs, deterministic=True
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break

            success = bool(info.get("success", False))
            successes += int(success)
            episode_rewards.append(total_reward)

            print(
                f"Episode {episode + 1}: "
                f"Reward={total_reward:.2f}, "
                f"Steps={steps}, "
                f"Success={success}"
            )            
            
            # Display why the episode ended
            failure_flags = {
                "jackknife": bool(info.get("jackknife", False)),
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "off_track": bool(info.get("off_track", False)),
            }

            print(
                f"  Terminated={terminated}, "
                f"Truncated={truncated}, "
                f"Failure flags={failure_flags}"
            )


            # Inspect the underlying simulator
            sim = base_env.unwrapped

            print(
                f"  Distance failure: {sim.dist_too_large} "
                f"(error={sim.audit_error:.2f} m, "
                f"limit={sim.max_path_error:.2f} m)"
            )


            print(
                f"  Other simulator flags: "
                f"fin={sim.fin}, "
                f"times_up={sim.times_up}, "
                f"goal={sim.goal}"
            )

            print(
                f"  Path progress: "
                f"{sim.last_index}/{len(sim.track_vector)}"
            )


            print(
                f"  Heading failure: {sim.angle_too_large} "
                f"(error={np.degrees(abs(sim.s[1])):.2f} deg, "
                f"limit={np.degrees(sim.max_heading_error):.2f} deg)"
            )



        # 4. Summarize the evaluation
        print("\n--- PPO Evaluation Results ---")
        print(f"Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"Success rate: {100 * successes / n_episodes:.1f}%")

    finally:
        env.close()


if __name__ == "__main__":
    main()
