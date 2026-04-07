"""
PPO Components - Homework 2, Problem 1 (Solutions)

Based on CleanRL's PPO implementation:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import torch
import numpy as np
from typing import Tuple


# =============================================================================
# Part 1: Core RL Computations
# =============================================================================


def compute_returns(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Compute discounted returns for each timestep.

    Args:
        rewards: Tensor of shape (num_steps, num_envs)
        dones: Tensor of shape (num_steps, num_envs)
        gamma: Discount factor

    Returns:
        returns: Tensor of shape (num_steps, num_envs)

    Nuances to handle:
        - Episode boundaries: When done[t]=1, the episode ended at timestep t,
          meaning r_t is the last reward of that episode. The return at step t
          should include r_t but not any rewards from future timesteps/episodes.
          The (1-done) term "cuts off" future returns at episode boundaries.
        - Edge case gamma=0: With no discounting, each timestep's return equals
          just its immediate reward.
    """
    num_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros_like(rewards[0])

    for t in reversed(range(num_steps)):
        running_return = rewards[t] + gamma * running_return * (1 - dones[t])
        returns[t] = running_return

    return returns


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Tensor of shape (num_steps, num_envs)
        values: Tensor of shape (num_steps + 1, num_envs)
                (includes bootstrap value at the end)
        dones: Tensor of shape (num_steps, num_envs)
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: Tensor of shape (num_steps, num_envs)

    Nuances to handle:
        - Episode boundaries require two separate (1-done) multipliers:
          (a) next_value should be 0 when done[t]=1 (episode ended, no future value)
          (b) GAE should not propagate advantages from future steps when done[t]=1
        - The values tensor has one more element along dim 0 than rewards.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])

    for t in reversed(range(num_steps)):
        next_value = values[t + 1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae * (1 - dones[t])
        advantages[t] = gae

    return advantages


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.

    Args:
        advantages: Tensor of shape (batch_size,) containing advantages

    Returns:
        normalized: Tensor of shape (batch_size,) containing normalized advantages

    Nuances to handle:
        - Add a small epsilon (1e-8) to the standard deviation to prevent division
          by zero when all advantages are identical.
        - Use torch.std() with its default settings (Bessel's correction, i.e.
          dividing by N-1) for the standard deviation.
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + 1e-8)


# =============================================================================
# Part 2: Policy Distribution Functions
# =============================================================================


def discrete_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability of actions under a categorical distribution.

    Args:
        logits: Tensor of shape (batch_size, num_actions) containing unnormalized log probabilities
        actions: Tensor of shape (batch_size,) containing action indices

    Returns:
        log_probs: Tensor of shape (batch_size,) containing log probabilities

    Nuances to handle:
        - Use log_softmax (not softmax then log) for numerical stability with
          large logit differences.
        - When gathering action probabilities, make sure to select the correct
          action for each sample in the batch independently.
    """
    # Convert logits to log probabilities using log_softmax
    log_probs_all = torch.log_softmax(logits, dim=-1)
    # Select the log probability of the taken action
    log_probs = log_probs_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    return log_probs


def discrete_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of a categorical distribution.

    Args:
        logits: Tensor of shape (batch_size, num_actions) containing unnormalized log probabilities

    Returns:
        entropy: Tensor of shape (batch_size,) containing entropy for each distribution

    Nuances to handle:
        - Use log_softmax for the log probabilities (not log(softmax)) for numerical
          stability with extreme logits.
        - Return per-sample entropy (shape batch_size), not a scalar.
    """
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    # Entropy = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def gaussian_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of actions under a diagonal Gaussian distribution.

    Args:
        mean: Tensor of shape (batch_size, action_dim) containing means
        log_std: Tensor of shape (action_dim,) or (batch_size, action_dim) containing log standard deviations
        actions: Tensor of shape (batch_size, action_dim) containing continuous actions

    Returns:
        log_probs: Tensor of shape (batch_size,) containing log probabilities

    Nuances to handle:
        - For multi-dimensional actions, the total log probability is the sum of
          log probabilities over all action dimensions (independent Gaussians).
        - log_std may be 1D (shared across batch) or 2D (per-sample). Broadcasting
          handles this automatically.
    """
    std = torch.exp(log_std)
    # Log probability of Gaussian: -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2 * pi)
    var = std**2
    log_prob_per_dim = -0.5 * (
        (actions - mean) ** 2 / var + 2 * log_std + np.log(2 * np.pi)
    )
    # Sum over action dimensions
    log_probs = log_prob_per_dim.sum(dim=-1)
    return log_probs


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of a diagonal Gaussian distribution.

    Args:
        log_std: Tensor of shape (action_dim,) or (batch_size, action_dim) containing log standard deviations

    Returns:
        entropy: Scalar tensor (if 1D input) or tensor of shape (batch_size,) (if 2D input)

    Nuances to handle:
        - For multi-dimensional actions, sum entropy over all dimensions.
        - Handle both 1D input (return scalar) and 2D input (return per-sample).
          Check log_std.dim() to determine which case you're in.
    """
    # Entropy of Gaussian: 0.5 * (1 + log(2 * pi * sigma^2)) = 0.5 + 0.5 * log(2 * pi) + log(sigma)
    # For diagonal Gaussian, sum over dimensions
    entropy_per_dim = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
    if log_std.dim() == 1:
        return entropy_per_dim.sum()
    else:
        return entropy_per_dim.sum(dim=-1)


# =============================================================================
# Part 2b: Action Sampling (Discrete and Continuous)
# =============================================================================


def sample_discrete_action(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample actions from a categorical distribution and return log probabilities.
    """
    probs = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    log_probs = discrete_log_prob(logits, actions)
    return actions, log_probs


def sample_continuous_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample continuous actions with tanh squashing and compute log probabilities.
    """
    std = torch.exp(log_std)
    noise = torch.randn_like(mean)
    z = mean + std * noise  # Unbounded sample
    action = torch.tanh(z)  # Squash to [-1, 1]

    # Log prob with Jacobian correction
    log_prob_z = gaussian_log_prob(mean, log_std, z)
    log_prob_correction = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
    log_prob = log_prob_z - log_prob_correction

    return action, log_prob


def squashed_gaussian_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    squashed_action: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of a squashed (tanh-transformed) action.
    """
    # Invert tanh to get unbounded action
    action_clamped = squashed_action.clamp(-0.999, 0.999)
    z = torch.atanh(action_clamped)

    # Gaussian log prob in unbounded space
    log_prob_z = gaussian_log_prob(mean, log_std, z)

    # Jacobian correction for tanh
    log_prob_correction = torch.log(1 - squashed_action.pow(2) + 1e-6).sum(dim=-1)

    return log_prob_z - log_prob_correction


def clip_action(
    action: torch.Tensor, low: float = -1.0, high: float = 1.0
) -> torch.Tensor:
    """
    Clip continuous actions to a specified range.
    """
    return torch.clamp(action, low, high)


# =============================================================================
# Part 3: PPO Loss Functions
# =============================================================================


def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """
    Compute the clipped PPO policy loss.

    Args:
        log_probs: Tensor of shape (batch_size,) containing current log probabilities
        old_log_probs: Tensor of shape (batch_size,) containing old log probabilities
        advantages: Tensor of shape (batch_size,) containing advantages
        clip_epsilon: Clipping parameter epsilon

    Returns:
        loss: Scalar tensor containing the policy loss (negated for gradient descent)

    Nuances to handle:
        - The min() creates a "pessimistic bound" - we always take the worse estimate
          of the objective, whether the advantage is positive or negative.
        - With positive advantage: clipping prevents ratio from going too high
        - With negative advantage: clipping prevents ratio from going too low
        - Return the negative of the objective (we maximize objective but
          optimizers minimize loss).
    """
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # PPO objective: min(ratio * A, clipped_ratio * A)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    # Take the minimum (pessimistic bound)
    policy_objective = torch.min(surrogate1, surrogate2)

    # Return negative because we want to maximize the objective but optimizers minimize
    return -policy_objective.mean()


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the value function loss (mean squared error).

    Args:
        values: Tensor of shape (batch_size,) containing value predictions
        returns: Tensor of shape (batch_size,) containing target returns

    Returns:
        loss: Scalar tensor containing the value loss

    """
    return ((values - returns) ** 2).mean()


def compute_entropy_bonus(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a categorical distribution for entropy bonus.

    Args:
        probs: Tensor of shape (batch_size, num_actions) containing action probabilities
               (i.e., after softmax). Unlike discrete_entropy which takes raw logits and
               applies softmax internally, this function expects pre-normalized probabilities
               that sum to 1 along the last dimension.

    Returns:
        entropy: Scalar tensor containing the mean entropy across the batch

    Nuances to handle:
        - Add epsilon (1e-8) before taking log to handle the case where a probability
          is exactly 0. Without this, log(0) = -inf and you'll get NaN.
        - Return the mean entropy across the batch, not the sum.
    """
    # Entropy = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()


# =============================================================================
# Part 4: Rollout Buffer for Vectorized Environments
# =============================================================================


class RolloutBuffer:
    """
    Buffer for storing trajectories from vectorized environments.

    Handles partial episodes correctly by bootstrapping with value estimates
    when the rollout ends before episode termination.

    Key invariant: returns = advantages + values
        This comes from the definition advantage = return - value. If your GAE
        computation is correct, this identity must hold.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the rollout buffer.

        Args:
            num_steps: Number of steps to collect per rollout
            num_envs: Number of parallel environments
            obs_shape: Shape of observations
            action_shape: Shape of actions (empty tuple for discrete)
            device: Device to store tensors on
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Storage tensors
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

        self.step = 0

    def reset(self):
        """Reset the buffer for a new rollout."""
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Add a transition to the buffer.

        Args:
            obs: Observation tensor of shape (num_envs,) + obs_shape
            action: Action tensor of shape (num_envs,) + action_shape
            log_prob: Log probability tensor of shape (num_envs,)
            reward: Reward tensor of shape (num_envs,)
            done: Done flag tensor of shape (num_envs,)
            value: Value estimate tensor of shape (num_envs,)
        """
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and GAE advantages for the collected rollout.

        This correctly handles partial episodes by using the bootstrap value
        when the rollout ends before the episode terminates.

        Args:
            last_value: Value estimate for the state after the last step, shape (num_envs,)
            last_done: Done flags for the last step, shape (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            returns: Tensor of shape (num_steps, num_envs)
            advantages: Tensor of shape (num_steps, num_envs)
        """
        # Build extended values tensor: (num_steps + 1, num_envs)
        # Bootstrap value is zeroed when last_done=1 (episode ended)
        bootstrap = (last_value * (1 - last_done)).unsqueeze(0)  # (1, num_envs)
        all_values = torch.cat([self.values, bootstrap], dim=0)

        # compute_gae handles the full (num_steps, num_envs) tensors at once
        advantages = compute_gae(
            rewards=self.rewards,
            values=all_values,
            dones=self.dones,
            gamma=gamma,
            lam=gae_lambda,
        )

        returns = advantages + self.values
        return returns, advantages

    def get_batches(
        self, batch_size: int, returns: torch.Tensor, advantages: torch.Tensor
    ):
        """
        Generate random minibatches for training.

        Args:
            batch_size: Size of each minibatch
            returns: Computed returns tensor
            advantages: Computed advantages tensor

        Yields:
            Dictionary containing batch data
        """
        # Flatten the data
        total_size = self.num_steps * self.num_envs
        indices = torch.randperm(total_size)

        # Reshape tensors
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        b_log_probs = self.log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = self.values.reshape(-1)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                "obs": b_obs[batch_indices],
                "actions": b_actions[batch_indices],
                "log_probs": b_log_probs[batch_indices],
                "returns": b_returns[batch_indices],
                "advantages": b_advantages[batch_indices],
                "values": b_values[batch_indices],
            }
