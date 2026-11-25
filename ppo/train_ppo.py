
from pathlib import Path

import gymnasium as gym
import gym_truck_backerupper  # Registers the simulator with Gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from gym_truck_backerupper.mdp.wrap_env import (
    GymBackerUpperBackend,
    TruckMDPEnv,
)


def main():
    # Create the truck-trailer simulator
    base_env = gym.make("TruckBackerUpper-v0")

    # Connect our MDP in reverse-driving mode
    backend = GymBackerUpperBackend(base_env, velocity=-2.012)
    env = TruckMDPEnv(backend)

    # Track episode rewards and lengths
    env = Monitor(env)

    # Create the PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
        verbose=1,
        seed=42,
    )

    try:
        # Run a short training test
        model.learn(
            total_timesteps=50_000,
            log_interval=10
        )

        # Save the trained model
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(model_dir / "ppo_reverse_50k_terminalfix"))
        print("PPO smoke test completed successfully.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
