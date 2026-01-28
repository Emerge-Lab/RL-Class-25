"""
Script to build the transition table for MountainCar Q-iteration.
Run this once to generate the transition files for students.

WARNING: Students should NOT run this script or modify the transition tables.
The .npy files are provided as-is and will be used by the grading server.
If you regenerate them, your solution may not work correctly during evaluation.
"""

import numpy as np
from tqdm import tqdm
import gymnasium as gym


N_BINS = 200
N_ACTIONS = 3  # 0: push left, 1: no push, 2: push right

# MountainCar observation bounds
STATE_BOUNDS = [
    (-1.2, 0.6),    # Position
    (-0.07, 0.07),  # Velocity
]


def discretize_state(observation):
    if observation.ndim > 1:
        observation = observation[0]
    indices = []
    for i, (low, high) in enumerate(STATE_BOUNDS):
        val = np.clip(observation[i], low, high)
        scaled = (val - low) / (high - low) * (N_BINS - 1)
        idx = int(np.clip(np.round(scaled), 0, N_BINS - 1))
        indices.append(idx)
    return tuple(indices)


def undiscretize_state(indices):
    obs = []
    for i, (low, high) in enumerate(STATE_BOUNDS):
        val = low + (indices[i] / (N_BINS - 1)) * (high - low)
        obs.append(val)
    return np.array(obs)


def build_transition_table():
    print("Building transition table for MountainCar...")
    env = gym.make("MountainCar-v0")

    # Shape: (position_bins, velocity_bins, actions, next_state_dims)
    next_states = np.zeros((N_BINS, N_BINS, N_ACTIONS, 2), dtype=np.int32)
    rewards = np.zeros((N_BINS, N_BINS, N_ACTIONS), dtype=np.float32)
    dones = np.zeros((N_BINS, N_BINS, N_ACTIONS), dtype=bool)

    for s0 in tqdm(range(N_BINS), desc="Building transitions"):
        for s1 in range(N_BINS):
            continuous_state = undiscretize_state((s0, s1))

            for action in range(N_ACTIONS):
                # Reset and set state directly
                env.reset()
                env.unwrapped.state = continuous_state.copy()

                obs, reward, terminated, truncated, _ = env.step(action)
                next_state_indices = discretize_state(obs)

                next_states[s0, s1, action] = next_state_indices
                rewards[s0, s1, action] = reward
                dones[s0, s1, action] = terminated  # Don't include truncated

    env.close()

    np.save("transition_next_states.npy", next_states)
    np.save("transition_rewards.npy", rewards)
    np.save("transition_dones.npy", dones)
    print("Saved: transition_next_states.npy, transition_rewards.npy, transition_dones.npy")

    # Print some stats
    print(f"\nTransition table stats:")
    print(f"  Shape P: {next_states.shape}")
    print(f"  Shape R: {rewards.shape}")
    print(f"  Shape D: {dones.shape}")
    print(f"  Terminal states: {dones.sum()} / {dones.size} ({100*dones.sum()/dones.size:.1f}%)")
    print(f"  Reward range: [{rewards.min()}, {rewards.max()}]")


if __name__ == "__main__":
    build_transition_table()
