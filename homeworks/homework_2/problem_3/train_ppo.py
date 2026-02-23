"""
Homework 2, Problem 3: PPO Training on Pong

Train a PPO agent on PufferLib's native Pong environment using the components
you implemented in Problem 1.

Your task: Implement the `train()` function below. Everything else is provided.

Usage:
    uv run python train_ppo.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.ocean.pong.pong import Pong

from homeworks.homework_2.problem_1.ppo_components import (  # noqa: F401
    RolloutBuffer,
    compute_entropy_bonus,
    compute_policy_loss,
    compute_value_loss,
    discrete_log_prob,
    normalize_advantages,
    sample_discrete_action,
)


# =============================================================================
# Hyperparameters (tuned — you shouldn't need to change these)
# =============================================================================
NUM_ENVS = 8
NUM_STEPS = 128  # steps per env per rollout
TOTAL_TIMESTEPS = 500_000
LR = 2.5e-4
NUM_EPOCHS = 4  # PPO update epochs per rollout
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Network Architecture (provided)
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

        # Orthogonal init (standard for PPO)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Smaller init for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Smaller init for value head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


# =============================================================================
# Evaluation (provided)
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
# Plotting (provided)
# =============================================================================
def plot_learning_curve(rewards, filename="ppo_pong.png"):
    """Save a learning curve plot."""
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
# YOUR TASK: Implement train()
# =============================================================================
def train():
    """
    Train a PPO agent on Pong and return a list of mean episode rewards per rollout.

    High-level steps:
        1. Create the Pong environment (with max_score=5), ActorCritic model, optimizer, and RolloutBuffer.
        2. Collect rollouts: for NUM_STEPS, use the model to get logits and values,
           sample actions with sample_discrete_action, step the environment, and
           store transitions in the buffer.
        3. After each rollout, compute advantages and returns using the buffer's
           compute_returns_and_advantages method (you need the value of the last
           observation as the bootstrap value).
        4. PPO update: for NUM_EPOCHS, iterate over minibatches from the buffer.
           For each minibatch:
             - Run the model on the batch observations to get new logits and values
             - Compute new log probs with discrete_log_prob
             - Normalize advantages
             - Compute policy loss (compute_policy_loss)
             - Compute value loss (compute_value_loss)
             - Compute entropy bonus (compute_entropy_bonus) using softmax(logits)
             - Total loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
             - Backprop, clip gradients (max_norm=MAX_GRAD_NORM), optimizer step
        5. Track mean episode rewards for each rollout.

    Returns:
        List of mean episode rewards (one per rollout).
    """
    raise NotImplementedError("Implement train()")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    # Load the final model for evaluation
    model = ActorCritic(8, 3).to(DEVICE)
    model.load_state_dict(torch.load("ppo_pong.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.2f}")
