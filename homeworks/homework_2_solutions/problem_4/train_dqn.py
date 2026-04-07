"""
Homework 2, Problem 4: DQN Training on CartPole (Solution)

Train a DQN agent on CartPole-v1 using the components you implemented in
Problem 2.

NOTE: DQN is often not a very performant algorithm compared to policy gradient
methods like PPO. We have you implement it because its *components* — replay
buffers, target networks, epsilon-greedy exploration, and TD learning — show up
throughout modern RL. Think of this as learning the building blocks, not as the
state-of-the-art way to solve environments.

Usage:
    uv run python train_dqn.py
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib
import gymnasium as gym

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homeworks.homework_2_solutions.problem_2.dqn_components import (
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
# Hyperparameters
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
# Evaluation
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
# Plotting
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
# Training
# =============================================================================
def train():
    """Train DQN on CartPole-v1."""
    # --- Setup ---
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]  # 4
    act_dim = env.action_space.n  # 2

    online_net = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    target_net = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    hard_update(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    replay_buffer = NStepReplayBuffer(BUFFER_CAPACITY, n_step=N_STEP, gamma=GAMMA)

    episode_rewards = []
    ep_reward = 0.0

    for global_step in range(TOTAL_TIMESTEPS):
        # --- Select action ---
        epsilon = linear_epsilon_decay(
            global_step, EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS
        )

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = online_net(obs_t)

        action = epsilon_greedy_action(q_values[0], epsilon, num_actions=act_dim)

        # --- Step environment ---
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        # --- Store transition ---
        replay_buffer.push(
            state=obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done,
        )

        ep_reward += reward
        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # --- Train ---
        if global_step >= LEARNING_STARTS and global_step % TRAIN_FREQ == 0:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions_b, rewards_b, next_states, dones = batch_to_tensors(
                batch, DEVICE
            )

            td_targets = compute_double_dqn_target(
                rewards=rewards_b,
                next_states=next_states,
                dones=dones,
                # Use GAMMA**N_STEP because the n-step buffer already
                # discounted the first N_STEP rewards internally.
                gamma=GAMMA**N_STEP,
                online_network=online_net,
                target_network=target_net,
            )

            q_vals = online_net(states)
            loss = compute_td_loss(q_vals, actions_b, td_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Update target network ---
        if global_step % TARGET_UPDATE_FREQ == 0:
            hard_update(online_net, target_net)

        # --- Logging ---
        if len(episode_rewards) > 0 and global_step % 5000 == 0:
            recent = (
                episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            )
            print(
                f"Step {global_step} | "
                f"Episodes: {len(episode_rewards)} | "
                f"Mean reward (last 50): {np.mean(recent):.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    # Save model
    torch.save(online_net.state_dict(), "dqn_cartpole.pt")
    print("Model saved to dqn_cartpole.pt")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    obs_dim = 4
    act_dim = 2
    model = QNetwork(state_dim=obs_dim, action_dim=act_dim).to(DEVICE)
    model.load_state_dict(torch.load("dqn_cartpole.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.1f}")
