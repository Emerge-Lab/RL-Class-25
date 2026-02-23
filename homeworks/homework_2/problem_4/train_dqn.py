"""
Homework 2, Problem 4: DQN Training on CartPole

Train a DQN agent on CartPole-v1 using the components you implemented in
Problem 2.

NOTE: DQN is often not a very performant algorithm compared to policy gradient
methods like PPO. We have you implement it because its *components* — replay
buffers, target networks, epsilon-greedy exploration, and TD learning — show up
throughout modern RL. Think of this as learning the building blocks, not as the
state-of-the-art way to solve environments.

Your task: Implement the `train()` function below. Everything else is provided.

Usage:
    uv run python train_dqn.py
"""

import numpy as np
import torch
import matplotlib
import gymnasium as gym

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homeworks.homework_2.problem_2.dqn_components import (  # noqa: F401
    QNetwork,
    NStepReplayBuffer,
    batch_to_tensors,
    compute_double_dqn_target,
    compute_td_loss,
    epsilon_greedy_action,
    hard_update,
    linear_epsilon_decay,
)

# =============================================================================
# Hyperparameters (tuned — you shouldn't need to change these)
# =============================================================================
TOTAL_TIMESTEPS = 200_000
LR = 1e-3
BATCH_SIZE = 128
BUFFER_CAPACITY = 10_000
GAMMA = 0.99
N_STEP = 3  # n-step returns for better credit assignment
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10_000
TARGET_UPDATE_FREQ = 500  # steps between hard target updates
LEARNING_STARTS = 1_000  # fill buffer before training
TRAIN_FREQ = 4  # train every N env steps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Evaluation (provided)
# =============================================================================
def evaluate(model, num_episodes=10):
    """Run greedy policy and return mean episode reward."""
    env = gym.make("CartPole-v1")
    rewards_total = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_vals = model(obs_t)
            action = q_vals.argmax(dim=-1).item()
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc

        rewards_total.append(ep_reward)

    env.close()
    return np.mean(rewards_total)


# =============================================================================
# Plotting (provided)
# =============================================================================
def plot_learning_curve(rewards, filename="dqn_cartpole.png"):
    """Save a learning curve plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    if len(rewards) >= 50:
        smooth = np.convolve(rewards, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(rewards)), smooth, label="50-episode avg")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Learning curve saved to {filename}")


# =============================================================================
# YOUR TASK: Implement train()
# =============================================================================
def train():
    """
    Train a DQN agent on CartPole-v1 and return a list of episode rewards.

    High-level steps:
        1. Create a CartPole-v1 environment (gym.make("CartPole-v1")),
           online/target QNetworks, optimizer, and NStepReplayBuffer
           (with N_STEP and GAMMA).
           - obs_dim = env.observation_space.shape[0]  (4)
           - act_dim = env.action_space.n  (2)
        2. Initialize the target network as a copy of the online network
           using hard_update(online_net, target_net).
        3. Step loop (TOTAL_TIMESTEPS iterations):
           a. Compute epsilon using linear_epsilon_decay.
           b. Get Q-values from the online network (unsqueeze obs to add
              batch dim), then select an action with epsilon_greedy_action.
           c. Step the environment. Push (obs, action, reward, next_obs, done)
              to the replay buffer.
           d. Track episode rewards: accumulate rewards, and when an episode
              ends (term or trunc), record the total and reset.
              Don't forget to call env.reset() when done!
           e. After LEARNING_STARTS steps, train every TRAIN_FREQ steps:
              - Sample a batch from the replay buffer
              - Convert to tensors with batch_to_tensors
              - Compute targets with compute_double_dqn_target.
                IMPORTANT: pass gamma=GAMMA**N_STEP (not GAMMA) because the
                n-step buffer already discounted the first N_STEP rewards.
              - Compute loss with compute_td_loss
              - Backprop and optimizer step
           f. Hard-update the target network every TARGET_UPDATE_FREQ steps.
        4. Save the model (torch.save) and return episode rewards.

    Returns:
        List of episode rewards (one per completed episode).
    """
    raise NotImplementedError("Implement train()")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    # Load the final model for evaluation
    obs_dim = 4
    act_dim = 2
    model = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    model.load_state_dict(torch.load("dqn_cartpole.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.1f}")
