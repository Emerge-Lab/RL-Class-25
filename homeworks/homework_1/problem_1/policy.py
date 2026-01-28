"""
Policy file for MountainCar Q-iteration submission.

This file will be used by the leaderboard to evaluate your trained agent.

Requirements:
    - Submit this file along with your checkpoint.pt (Q-table saved with torch.save)
    - Do NOT modify the discretization parameters or Policy class structure
"""

import numpy as np
import torch


# =============================================================================
# FIXED DISCRETIZATION PARAMETERS - DO NOT MODIFY
# =============================================================================
N_BINS = 200
STATE_BOUNDS = [
    (-1.2, 0.6),    # Position
    (-0.07, 0.07),  # Velocity
]


class Policy:
    def __init__(self):
        self.q_table = None

    def discretize_state(self, observation: np.ndarray) -> tuple:
        """
        Convert a continuous observation to discrete grid indices.
        DO NOT MODIFY THIS FUNCTION.
        """
        if observation.ndim > 1:
            observation = observation[0]

        indices = []
        for i, (low, high) in enumerate(STATE_BOUNDS):
            val = np.clip(observation[i], low, high)
            scaled = (val - low) / (high - low) * (N_BINS - 1)
            idx = int(np.clip(np.round(scaled), 0, N_BINS - 1))
            indices.append(idx)

        return tuple(indices)

    def forward(self, obs: np.ndarray) -> int:
        """
        Select an action given an observation.

        Args:
            obs: Current observation from the environment
                 Shape: (2,) or (1, 2) for single environment

        Returns:
            Action to take: 0 (push left), 1 (no push), or 2 (push right)

        TODO: Implement this method
            1. Discretize the observation using self.discretize_state(obs)
            2. Look up the Q-values for that state in self.q_table
            3. Return the action with the highest Q-value (as an int)

        Hint: Use np.argmax to find the action with the highest Q-value
        """
        raise NotImplementedError("Implement the forward method")


def load_policy(checkpoint_path: str) -> Policy:
    """
    Load a trained policy from a Q-table checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint.pt file (Q-table saved with torch.save)

    Returns:
        Policy instance with loaded Q-table
    """
    policy = Policy()
    policy.q_table = torch.load(checkpoint_path, weights_only=True)
    if isinstance(policy.q_table, torch.Tensor):
        policy.q_table = policy.q_table.numpy()
    assert policy.q_table.shape == (N_BINS, N_BINS, 3), \
        f"Q-table must have shape (200, 200, 3), got {policy.q_table.shape}"
    return policy


# For testing the policy locally
if __name__ == "__main__":
    import gymnasium as gym

    policy = load_policy("checkpoint.pt")
    env = gym.make("MountainCar-v0", render_mode="human")
    obs, _ = env.reset(seed=42)

    total_reward = 0
    done = False
    steps = 0

    print("Testing policy...")
    while not done and steps < 200:
        action = policy.forward(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    if terminated:
        print(f"Success! Reached goal in {steps} steps with reward {total_reward}")
    else:
        print(f"Failed to reach goal in {steps} steps with reward {total_reward}")
    env.close()
