"""
Problem 1: Tabular Q-Iteration for MountainCar - SOLUTION
"""

import numpy as np
import torch
import gymnasium as gym


# =============================================================================
# FIXED PARAMETERS
# =============================================================================
N_BINS = 200
N_ACTIONS = 3
STATE_BOUNDS = [
    (-1.2, 0.6),  # Position
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
    """Load the precomputed transition tables."""
    import os

    dir_path = os.path.dirname(os.path.abspath(__file__))
    hw_path = os.path.join(dir_path, "..", "..", "homework_1", "problem_1")
    P = np.load(os.path.join(hw_path, "transition_next_states.npy"))
    R = np.load(os.path.join(hw_path, "transition_rewards.npy"))
    D = np.load(os.path.join(hw_path, "transition_dones.npy"))
    return P, R, D


# =============================================================================
# SOLUTION
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
    """
    # Initialize Q-table to zeros
    Q = np.zeros((N_BINS, N_BINS, N_ACTIONS))

    for iteration in range(max_iterations):
        Q_new = np.zeros_like(Q)

        # For each state and action, compute the Bellman update
        for s0 in range(N_BINS):
            for s1 in range(N_BINS):
                for a in range(N_ACTIONS):
                    # Get next state indices
                    ns = P[s0, s1, a]
                    ns0, ns1 = ns[0], ns[1]

                    # Get reward and done flag
                    r = R[s0, s1, a]
                    d = D[s0, s1, a]

                    # Bellman update
                    if d:
                        Q_new[s0, s1, a] = r
                    else:
                        Q_new[s0, s1, a] = r + gamma * np.max(Q[ns0, ns1])

        # Check convergence
        delta = np.max(np.abs(Q_new - Q))
        Q = Q_new

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: delta = {delta:.8f}")

        if delta < theta:
            print(f"Converged after {iteration + 1} iterations (delta = {delta:.8f})")
            break

    return Q


def q_iteration_vectorized(
    P: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> np.ndarray:
    """
    Vectorized version of Q-iteration.
    """
    Q = np.zeros((N_BINS, N_BINS, N_ACTIONS))

    # Precompute indices for gathering next state Q-values
    idx0 = P[..., 0]  # shape (N_BINS, N_BINS, 3)
    idx1 = P[..., 1]

    for iteration in range(max_iterations):
        # Get max Q-value for each state
        V = np.max(Q, axis=-1)  # shape (N_BINS, N_BINS)

        # Gather V(s') for each (s, a) pair
        V_next = V[idx0, idx1]  # shape (N_BINS, N_BINS, 3)

        # Bellman update
        Q_new = R + gamma * (1 - D.astype(float)) * V_next

        # Check convergence
        delta = np.max(np.abs(Q_new - Q))
        Q = Q_new

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: delta = {delta:.8f}")

        if delta < theta:
            print(f"Converged after {iteration + 1} iterations (delta = {delta:.8f})")
            break

    return Q


# =============================================================================
# EVALUATION AND SAVING
# =============================================================================


def evaluate_discrete(
    q_table: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    num_episodes: int = 100,
    max_steps: int = 200,
) -> float:
    """
    Evaluate the learned Q-table on the discretized MDP.

    This evaluates using the same transition model used for training,
    ensuring consistency between training and evaluation.
    """
    # Starting states: position in [-0.6, -0.4], velocity 0
    # For N_BINS=200: position bins ~66-88, velocity bin 100
    pos_low, pos_high = STATE_BOUNDS[0]
    vel_low, vel_high = STATE_BOUNDS[1]
    start_pos_min = int(round((-0.6 - pos_low) / (pos_high - pos_low) * (N_BINS - 1)))
    start_pos_max = int(round((-0.4 - pos_low) / (pos_high - pos_low) * (N_BINS - 1)))
    start_vel_bin = int(round((0.0 - vel_low) / (vel_high - vel_low) * (N_BINS - 1)))
    start_pos_bins = list(range(start_pos_min, start_pos_max + 1))

    total_rewards = []
    successes = 0

    for episode in range(num_episodes):
        # Random starting position within the valid range
        s0 = np.random.choice(start_pos_bins)
        s1 = start_vel_bin

        episode_reward = 0

        for step in range(max_steps):
            # Select action greedily
            action = int(np.argmax(q_table[s0, s1]))

            # Get reward and check if done
            reward = R[s0, s1, action]
            done = D[s0, s1, action]
            episode_reward += reward

            if done:
                successes += 1
                break

            # Transition to next state
            next_state = P[s0, s1, action]
            s0, s1 = next_state[0], next_state[1]

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(
        f"[Discrete MDP] Average reward over {num_episodes} episodes: {avg_reward:.2f}"
    )
    print(
        f"[Discrete MDP] Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)"
    )
    return avg_reward


def evaluate_gym(
    q_table: np.ndarray, num_episodes: int = 100, render: bool = False
) -> float:
    """Evaluate the learned Q-table in the continuous gym environment."""
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
    print(f"[Gym Env] Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(
        f"[Gym Env] Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)"
    )
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
    print("\nRunning Q-iteration (vectorized)...")
    q_table = q_iteration_vectorized(P, R, D, gamma=0.99)

    # Save the Q-table
    save_q_table(q_table)

    # Evaluate on discretized MDP (should be 100% success)
    print("\nEvaluating on discrete MDP...")
    evaluate_discrete(q_table, P, R, D, num_episodes=100)

    # Evaluate on continuous gym environment
    print("\nEvaluating on continuous gym environment...")
    evaluate_gym(q_table, num_episodes=100)
    evaluate_gym(q_table, num_episodes=2, render=True)
