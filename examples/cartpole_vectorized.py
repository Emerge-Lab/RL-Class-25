"""Example of running PufferLib's native CartPole with multiple parallel environments."""

import numpy as np
from pufferlib.ocean.cartpole.cartpole import Cartpole


def main():
    num_envs = 16  # Run 16 environments in parallel

    # Create vectorized CartPole environment
    # All environments step simultaneously for high throughput
    env = Cartpole(num_envs=num_envs, render_mode=None)

    # Reset all environments
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")  # (num_envs, 4)
    print(f"Running {num_envs} parallel environments")

    total_rewards = np.zeros(num_envs)
    episode_counts = np.zeros(num_envs, dtype=int)

    for step in range(100):
        # Sample random actions for all environments
        actions = np.array([env.single_action_space.sample() for _ in range(num_envs)])

        # Step all environments simultaneously
        obs, rewards, terminals, truncations, info = env.step(actions)
        total_rewards += rewards

        # Track episode completions per environment
        dones = terminals | truncations
        episode_counts += dones.astype(int)

    print(f"\nCompleted 100 steps across {num_envs} environments")
    print(f"Total episodes completed: {episode_counts.sum()}")
    print(f"Episodes per env: {episode_counts}")
    print(f"Total reward per env: {total_rewards}")
    print(f"Mean reward: {total_rewards.mean():.1f}")

    env.close()


if __name__ == "__main__":
    main()
