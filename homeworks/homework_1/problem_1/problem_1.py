"""
Problem 1: Tabular Q-Iteration for MountainCar

In this problem, you will implement Q-iteration (dynamic programming) for MountainCar.
The continuous observation space has been discretized into a 200x200 grid.

The MountainCar observation space has 2 dimensions:
    0: Position  (range: -1.2 to 0.6, goal at position >= 0.5)
    1: Velocity  (range: -0.07 to 0.07)

Actions:
    0: Push left
    1: No push
    2: Push right

We provide precomputed transition tables that describe the environment dynamics.
WARNING: Do NOT modify or regenerate the transition tables (.npy files).
The grading server uses the same tables - changing them will cause your submission to fail.

TRANSITION TABLES:
=================

The state space is discretized into a 200x200 grid:
    - s0 (first index): Position index, ranging from 0 to 199
        - Index 0 corresponds to position -1.2 (leftmost)
        - Index 199 corresponds to position 0.6 (rightmost)
        - The goal region (position >= 0.5) is roughly indices 189-199
    - s1 (second index): Velocity index, ranging from 0 to 199
        - Index 0 corresponds to velocity -0.07 (moving left fastest)
        - Index 199 corresponds to velocity 0.07 (moving right fastest)
        - Index ~100 corresponds to velocity ~0 (stationary)

The transition tables are numpy arrays:

    P (transition_next_states.npy): shape (200, 200, 3, 2)
        P[s0, s1, a] = [s0', s1']  (a numpy array of 2 integers)

        Given current state indices (s0, s1) and action a, returns the
        next state indices [s0', s1'].

        Example:
            next_state = P[100, 100, 2]  # State (100,100), action 2 (push right)
            s0_next, s1_next = next_state[0], next_state[1]
            # Now you can look up Q[s0_next, s1_next, :] to get Q-values at next state

    R (transition_rewards.npy): shape (200, 200, 3)
        R[s0, s1, a] = reward (a single float)

        The immediate reward for taking action a in state (s0, s1).
        In MountainCar, this is always -1 for every step (encourages reaching
        the goal quickly).

        Example:
            reward = R[100, 100, 2]  # Always -1.0

    D (transition_dones.npy): shape (200, 200, 3)
        D[s0, s1, a] = done (a boolean: True or False)

        Whether the episode terminates after taking action a in state (s0, s1).
        True only when the car reaches the goal (position >= 0.5).

        Example:
            done = D[195, 150, 2]  # True if this transition reaches the goal
            # If done is True, there is no future reward (episode ends)

USING THE TABLES IN Q-ITERATION:
===============================

For each state-action pair (s0, s1, a):
    1. Look up the next state: s0', s1' = P[s0, s1, a]
    2. Look up the reward: r = R[s0, s1, a]
    3. Look up if terminal: d = D[s0, s1, a]
    4. Apply Bellman update:
       Q_new[s0, s1, a] = r + gamma * (1 - d) * max_a' Q[s0', s1', a']

       Note: (1 - d) ensures we don't add future value for terminal states

Your task:
    Implement Q-iteration using the Bellman optimality equation:

        Q(s, a) = R(s, a) + gamma * (1 - D(s, a)) * max_a' Q(s', a')

    where s' = P(s, a) is the next state (deterministic in MountainCar).

Submission:
    - checkpoint.pt: Your Q-table saved with torch.save (shape: 200x200x3)
    - policy.py: Implement the forward() method to select actions using your Q-table

Expected performance:
    A well-implemented solution should consistently reach the goal.
    The episode terminates when position >= 0.5 (success) or after 200 steps (failure).
    Success means reaching the goal; the total reward will be negative (fewer steps = better).

    Your policy must achieve a mean reward better than -150 to pass.
"""

import numpy as np
import torch
import gymnasium as gym


# =============================================================================
# FIXED PARAMETERS - DO NOT MODIFY
# =============================================================================
N_BINS = 200
N_ACTIONS = 3
STATE_BOUNDS = [
    (-1.2, 0.6),    # Position
    (-0.07, 0.07),  # Velocity
]


def discretize_state(observation: np.ndarray) -> tuple:
    """Convert a continuous observation to discrete grid indices."""
    if observation.ndim > 1:
        observation = observation[0]
    indices = []
    for i, (low, high) in enumerate(STATE_BOUNDS):
        val = np.clip(observation[i], low, high)
        scaled = (val - low) / (high - low) * (N_BINS - 1)
        idx = int(np.clip(np.round(scaled), 0, N_BINS - 1))
        indices.append(idx)
    return tuple(indices)


def load_transition_tables():
    """
    Load the precomputed transition tables.

    Returns:
        P: next state table, shape (200, 200, 3, 2)
           P[s0, s1, a] gives the next state indices [s0', s1']
        R: reward table, shape (200, 200, 3)
           R[s0, s1, a] gives the reward for taking action a in state s
        D: done table, shape (200, 200, 3)
           D[s0, s1, a] is True if the episode terminates (goal reached)
    """
    import os
    dir_path = os.path.dirname(os.path.abspath(__file__))
    P = np.load(os.path.join(dir_path, "transition_next_states.npy"))
    R = np.load(os.path.join(dir_path, "transition_rewards.npy"))
    D = np.load(os.path.join(dir_path, "transition_dones.npy"))
    return P, R, D


# =============================================================================
# TODO: IMPLEMENT THE FUNCTION BELOW
# =============================================================================


def q_iteration(
    P: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> np.ndarray:
    """
    Perform Q-iteration (value iteration on Q-function).

    The Bellman optimality equation for Q-values:
        Q(s, a) = R(s, a) + gamma * (1 - D(s, a)) * max_a' Q(s', a')

    where s' = P(s, a) is deterministic in MountainCar.

    Args:
        P: Transition table, shape (200, 200, 3, 2)
           P[s0, s1, a] = [s0', s1'] (next state indices)
        R: Reward table, shape (200, 200, 3)
        D: Done table, shape (200, 200, 3)
        gamma: Discount factor
        theta: Convergence threshold (stop when max|Q_new - Q| < theta)
        max_iterations: Maximum number of iterations

    Returns:
        Converged Q-table of shape (200, 200, 3)

    TODO: Implement Q-iteration
        1. Initialize Q-table to zeros: shape (200, 200, 3)
        2. Repeat until convergence or max_iterations:
            - For each state (s0, s1) and action a:
                - Look up next state: s' = P[s0, s1, a]
                - Look up reward: r = R[s0, s1, a]
                - Look up done: d = D[s0, s1, a]
                - Compute: Q_new[s, a] = r + gamma * (1 - d) * max_a' Q[s', a']
            - Compute delta = max|Q_new - Q|
            - If delta < theta: converged, stop
            - Q = Q_new
        3. Return the converged Q-table

    Hint: You can vectorize this for efficiency, but a loop-based
    implementation is also fast for this small state space.
    """
    raise NotImplementedError("Implement q_iteration")


# =============================================================================
# EVALUATION AND SAVING - DO NOT MODIFY
# =============================================================================


def evaluate(q_table: np.ndarray, num_episodes: int = 100, render: bool = False) -> float:
    """Evaluate the learned Q-table."""
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)

    total_rewards = []
    successes = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_indices = discretize_state(obs)
            action = int(np.argmax(q_table[state_indices]))
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            if terminated:
                successes += 1

        total_rewards.append(episode_reward)

    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    return avg_reward


def save_q_table(q_table: np.ndarray, filepath: str = "checkpoint.pt") -> None:
    """Save the Q-table to a file using torch.save for server compatibility."""
    # Convert to tensor for safe loading with weights_only=True
    torch.save(torch.from_numpy(q_table), filepath)
    print(f"Q-table saved to {filepath}")


if __name__ == "__main__":
    # Load transition tables
    print("Loading transition tables...")
    P, R, D = load_transition_tables()
    print(f"  P (next states): {P.shape}")
    print(f"  R (rewards):     {R.shape}")
    print(f"  D (dones):       {D.shape}")

    # Run Q-iteration
    print("\nRunning Q-iteration...")
    q_table = q_iteration(P, R, D, gamma=0.99)

    # Save the Q-table
    save_q_table(q_table)

    # Evaluate
    print("\nEvaluating...")
    evaluate(q_table, num_episodes=100)

    # Watch the trained agent
    print("\nWatching trained agent...")
    evaluate(q_table, num_episodes=3, render=True)
