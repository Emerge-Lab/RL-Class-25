"""
Homework 2, Problem 3: PPO Training on Pong (Solution)

Usage:
    uv run python train_ppo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.ocean.pong.pong import Pong

from homeworks.homework_2_solutions.problem_1.ppo_components import (
    RolloutBuffer,
    compute_entropy_bonus,
    compute_policy_loss,
    compute_value_loss,
    discrete_log_prob,
    normalize_advantages,
    sample_discrete_action,
)

# =============================================================================
# Hyperparameters
# =============================================================================
NUM_ENVS = 8
NUM_STEPS = 128
TOTAL_TIMESTEPS = 500_000
LR = 2.5e-4
NUM_EPOCHS = 4
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Network
# =============================================================================
class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(model, num_episodes=10):
    """Run greedy policy and return mean episode reward."""
    env = Pong(num_envs=1, max_score=5)
    obs, _ = env.reset()
    rewards_total = []
    ep_reward = 0.0

    while len(rewards_total) < num_episodes:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits, _ = model(obs_t)
        action = logits.argmax(dim=-1).cpu().numpy()
        obs, rewards, terms, truncs, _ = env.step(action)
        ep_reward += rewards[0]
        if terms[0] or truncs[0]:
            rewards_total.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

    env.close()
    return np.mean(rewards_total)


# =============================================================================
# Plotting
# =============================================================================
def plot_learning_curve(rewards, filename="ppo_pong.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="per-rollout")
    if len(rewards) >= 10:
        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), smooth, label="10-rollout avg")
    plt.xlabel("Rollout")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO on Pong")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Learning curve saved to {filename}")


# =============================================================================
# Training
# =============================================================================
def train():
    """Train PPO on Pong."""
    # --- Setup ---
    env = Pong(num_envs=NUM_ENVS, max_score=5)
    obs, _ = env.reset()

    model = ActorCritic(obs_dim=8, act_dim=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    buffer = RolloutBuffer(
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
        obs_shape=(8,),
        action_shape=(),
        device=DEVICE,
    )

    num_rollouts = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)
    reward_history = []

    # Per-env episode reward tracking
    ep_rewards = np.zeros(NUM_ENVS)
    rollout_ep_rewards = []

    global_step = 0

    for rollout in range(num_rollouts):
        buffer.reset()

        # --- Collect rollout ---
        for step in range(NUM_STEPS):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits, values = model(obs_t)
                actions, log_probs = sample_discrete_action(logits)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, terms, truncs, _ = env.step(actions_np)

            # Store transition
            buffer.add(
                obs=obs_t,
                action=actions,
                log_prob=log_probs,
                reward=torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
                done=torch.tensor(
                    np.logical_or(terms, truncs), dtype=torch.float32, device=DEVICE
                ),
                value=values,
            )

            # Track episode rewards
            ep_rewards += rewards
            for i in range(NUM_ENVS):
                if terms[i] or truncs[i]:
                    rollout_ep_rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0.0

            obs = next_obs
            global_step += NUM_ENVS

        # --- Compute advantages ---
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            _, last_value = model(obs_t)
            # After a step, we don't know if the env just reset — assume not done
            last_done = torch.zeros(NUM_ENVS, device=DEVICE)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        # --- PPO update ---
        for epoch in range(NUM_EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE, returns, advantages):
                logits, values = model(batch["obs"])
                new_log_probs = discrete_log_prob(logits, batch["actions"].long())

                # Losses
                adv = normalize_advantages(batch["advantages"])
                policy_loss = compute_policy_loss(
                    new_log_probs, batch["log_probs"], adv, CLIP_EPSILON
                )
                value_loss = compute_value_loss(values, batch["returns"])
                probs = torch.softmax(logits, dim=-1)
                entropy = compute_entropy_bonus(probs)

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # --- Logging ---
        if rollout_ep_rewards:
            mean_reward = np.mean(rollout_ep_rewards)
            reward_history.append(mean_reward)
            print(
                f"Rollout {rollout + 1}/{num_rollouts} | "
                f"Step {global_step} | "
                f"Mean reward: {mean_reward:.2f} | "
                f"Episodes: {len(rollout_ep_rewards)}"
            )
            rollout_ep_rewards = []
        else:
            reward_history.append(0.0)

    # Save model
    torch.save(model.state_dict(), "ppo_pong.pt")
    print("Model saved to ppo_pong.pt")

    env.close()
    return reward_history


if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    model = ActorCritic(8, 3).to(DEVICE)
    model.load_state_dict(torch.load("ppo_pong.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.2f}")
