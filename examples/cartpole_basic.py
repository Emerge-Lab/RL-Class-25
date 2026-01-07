"""Basic example of running PufferLib's native CartPole for 100 steps."""

import numpy as np
from pufferlib.ocean.cartpole.cartpole import Cartpole


def main():
    # Create PufferLib's native CartPole environment
    # num_envs controls how many parallel environments to run
    env = Cartpole(num_envs=1, render_mode='human')

    # Reset the environment
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.single_action_space}")

    total_reward = 0
    episodes = 0

    for step in range(100):
        # Sample random action
        action = np.array([env.single_action_space.sample()])

        # Step the environment
        obs, rewards, terminals, truncations, info = env.step(action)
        total_reward += rewards.sum()
        env.render()

        if terminals.any() or truncations.any():
            episodes += 1
            print(f"Episode {episodes} finished at step {step + 1}")
            obs, info = env.reset()

    print(f"\nCompleted 100 steps")
    print(f"Total episodes: {episodes}")
    print(f"Total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
