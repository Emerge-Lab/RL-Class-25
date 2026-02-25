"""
PPO Components - Homework 2, Problem 1

Implement each of the following functions. Run `uv run python test_components.py`
to check your implementations.

Based on CleanRL's PPO implementation:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import torch
from typing import Tuple


# =============================================================================
# Part 1: Core RL Computations
# =============================================================================


def compute_returns(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Compute discounted returns (cumulative future rewards) for each timestep.

    The return at timestep t is the sum of all future rewards, discounted by gamma:
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    This can be computed recursively (starting from the end of the trajectory):
        G_T = r_T                           (last timestep)
        G_t = r_t + gamma * G_{t+1}         (all other timesteps)

    Episode boundaries: When done[t] = 1, the episode ended at timestep t,
    meaning r_t is the last reward of that episode. The return at step t
    should include r_t but not any rewards from future timesteps/episodes:
        G_t = r_t + gamma * G_{t+1} * (1 - done[t])

    Args:
        rewards: Tensor of shape (num_steps, num_envs)
        dones: Tensor of shape (num_steps, num_envs)
        gamma: Discount factor (typically 0.99)

    Returns:
        returns: Tensor of shape (num_steps, num_envs)

    Example:
        rewards = [1, 1, 1, 1], dones = [0, 0, 0, 0], gamma = 0.99
        G_3 = 1
        G_2 = 1 + 0.99 * 1 = 1.99
        G_1 = 1 + 0.99 * 1.99 = 2.9701
        G_0 = 1 + 0.99 * 2.9701 = 3.940399

    Nuances to handle:
        - Episode boundaries: When done[t]=1, the return at timestep t should not
          include any rewards from future timesteps. The (1-done) term "cuts off"
          future returns at episode boundaries.
        - Edge case gamma=0: With no discounting, each timestep's return equals
          just its immediate reward.

    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_returns")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE provides a way to estimate advantages that balances bias and variance.
    It uses a weighted average of n-step advantage estimates.

    The key insight is that we can write the advantage as:
        A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...

    where delta_t is the TD error (temporal difference error):
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    The TD error tells us: "How much better was this step than we expected?"
    - If delta_t > 0: we got more reward than expected (good action)
    - If delta_t < 0: we got less reward than expected (bad action)

    GAE can be computed recursively (starting from the end):
        A_T = delta_T
        A_t = delta_t + (gamma * lambda) * A_{t+1} * (1 - done[t])

    Episode boundaries: When done[t] = 1:
    - The next state's value should be 0 (episode ended)
    - Future advantages should not propagate back

    Args:
        rewards: Tensor of shape (num_steps, num_envs)
        values: Tensor of shape (num_steps + 1, num_envs).
                values[t] = V(s_t), and values[num_steps] is the bootstrap value
                for the state after the last step (used for partial episodes)
        dones: Tensor of shape (num_steps, num_envs)
        gamma: Discount factor (typically 0.99)
        lam: GAE lambda parameter (typically 0.95). Controls bias-variance tradeoff:
             - lambda=0: Use only 1-step TD error (high bias, low variance)
             - lambda=1: Use full Monte Carlo return (low bias, high variance)

    Returns:
        advantages: Tensor of shape (num_steps, num_envs)

    Nuances to handle:
        - Episode boundaries require two separate (1-done) multipliers:
          (a) next_value should be 0 when done[t]=1 (episode ended, no future value)
          (b) GAE should not propagate advantages from future steps when done[t]=1
        - The values tensor has one more element along dim 0 than rewards.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_gae")


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.

    Normalization helps stabilize training by ensuring advantages have
    consistent scale across different batches and environments.

    Formula:
        normalized = (advantages - mean) / (std + epsilon)

    where epsilon (1e-8) prevents division by zero.

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
    # TODO: Implement this
    raise NotImplementedError("Implement normalize_advantages")


# =============================================================================
# Part 2: Policy Distribution Functions
# =============================================================================


def discrete_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability of actions under a categorical (discrete) distribution.

    In RL with discrete actions, the policy network outputs logits (unnormalized
    log probabilities) for each action. We need to:
    1. Convert logits to a proper probability distribution (using softmax)
    2. Extract the probability of the action that was actually taken
    3. Return the log of that probability (for numerical stability)

    The log_softmax function combines steps 1 and 3 efficiently:
        log_softmax(logits) = log(softmax(logits))
                            = logits - log(sum(exp(logits)))
    Note that you can just use Pytorch's version for this btw. I'm just describing
    the detail here for edification because doing it the other way is numerically unstable.

    Args:
        logits: Tensor of shape (batch_size, num_actions) containing raw network outputs.
                These are not probabilities - they can be any real numbers.
        actions: Tensor of shape (batch_size,) containing action indices (integers).
                 Each value is in range [0, num_actions-1].

    Returns:
        log_probs: Tensor of shape (batch_size,) containing log P(action | state)

    Example:
        logits = [[1.0, 2.0, 3.0]]  # 3 actions
        actions = [2]               # took action 2
        # softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
        # log_prob = log(0.67) ≈ -0.40

    Nuances to handle:
        - Use log_softmax (not softmax then log) for numerical stability with
          large logit differences.
        - When gathering action probabilities, make sure to select the correct
          action for each sample in the batch independently. A common bug is
          selecting the same action index for all samples.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement discrete_log_prob")


def discrete_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of a categorical distribution.

    Entropy measures the "randomness" or "uncertainty" of a distribution:
        H(p) = -sum_i p_i * log(p_i)

    Properties:
    - Maximum entropy: uniform distribution (all actions equally likely)
    - Minimum entropy (0): deterministic distribution (one action has prob 1)

    In PPO, we add an entropy bonus to the loss to encourage exploration.

    Args:
        logits: Tensor of shape (batch_size, num_actions) containing unnormalized log probs

    Returns:
        entropy: Tensor of shape (batch_size,) containing entropy for each distribution

    Nuances to handle:
        - Use log_softmax for the log probabilities (not log(softmax)) for numerical
          stability with extreme logits.
        - Return per-sample entropy (shape batch_size), not a scalar.

    """
    # TODO: Implement this
    raise NotImplementedError("Implement discrete_entropy")


def gaussian_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of actions under a diagonal Gaussian distribution.

    For continuous action spaces, we model the policy as a Gaussian:
        pi(a|s) = N(a; mu(s), sigma(s))

    For a multivariate Gaussian with diagonal covariance (independent dimensions),
    the log probability is the sum of log probabilities for each dimension:

        log p(a) = sum_i [ -0.5 * ((a_i - mu_i) / sigma_i)^2 - log(sigma_i) - 0.5 * log(2*pi) ]

    We use log_std instead of std for numerical stability and to allow
    unconstrained optimization (std must be positive, but log_std can be any value).

    Args:
        mean: Tensor of shape (batch_size, action_dim) containing the mean of the Gaussian
        log_std: Tensor of shape (action_dim,) or (batch_size, action_dim) containing
                 log of standard deviation. std = exp(log_std)
        actions: Tensor of shape (batch_size, action_dim) containing the continuous actions

    Returns:
        log_probs: Tensor of shape (batch_size,) containing total log probability
                   (summed over action dimensions)

    Nuances to handle:
        - For multi-dimensional actions, the total log probability is the sum of
          log probabilities over all action dimensions (independent Gaussians).
        - log_std may be 1D (shared across batch) or 2D (per-sample). Broadcasting
          handles this automatically.

    """
    # TODO: Implement this
    raise NotImplementedError("Implement gaussian_log_prob")


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of a diagonal Gaussian distribution.

    For a Gaussian, entropy has a closed-form solution:
        H = 0.5 * log(2 * pi * e * sigma^2)

    For a diagonal Gaussian (independent dimensions), total entropy is the sum
    over dimensions.

    Note: Gaussian entropy depends only on the standard deviation, not the mean.

    Args:
        log_std: Tensor of shape (action_dim,) or (batch_size, action_dim)

    Returns:
        entropy: Scalar tensor (if 1D input) or tensor of shape (batch_size,) (if 2D input)

    Nuances to handle:
        - For multi-dimensional actions, sum entropy over all dimensions.
        - Handle both 1D input (return scalar) and 2D input (return per-sample).
          Check log_std.dim() to determine which case you're in.

    """
    # TODO: Implement this
    raise NotImplementedError("Implement gaussian_entropy")


# =============================================================================
# Part 2b: Action Sampling (Discrete and Continuous)
# =============================================================================


def sample_discrete_action(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample actions from a categorical distribution and return log probabilities.

    Given logits (unnormalized log probabilities), this function:
    1. Converts to probabilities via softmax
    2. Samples actions from the categorical distribution
    3. Computes log probabilities of the sampled actions

    Args:
        logits: Tensor of shape (batch_size, num_actions) containing unnormalized log probs

    Returns:
        actions: Tensor of shape (batch_size,) containing sampled action indices
        log_probs: Tensor of shape (batch_size,) containing log P(action)
    """
    # TODO: Implement this
    raise NotImplementedError("Implement sample_discrete_action")


def sample_continuous_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample continuous actions with tanh squashing and compute log probabilities.

    For continuous control, we sample from a Gaussian and then apply tanh to
    bound actions to [-1, 1]. This is called the "squashed Gaussian" policy.

    The sampling process:
        1. Sample z ~ N(mean, std)           # Unbounded Gaussian sample
        2. action = tanh(z)                   # Squash to [-1, 1]

    The log probability requires a Jacobian correction for the tanh transform.
    When you apply a deterministic transformation a = tanh(z) to a random variable z,
    the probability density changes according to the change-of-variables formula:

        p(a) = p(z) * |dz/da|

    Since da/dz = 1 - tanh(z)^2 = 1 - a^2, we have |dz/da| = 1 / (1 - a^2).
    Taking the log and summing over action dimensions:

        log p(a) = log p(z) - sum(log(1 - a^2))

    Intuitively, tanh "compresses" the tails of the Gaussian — many different z values
    near +/-inf all map to actions near +/-1. The Jacobian correction accounts for this
    compression: actions near the boundaries are more likely than the raw Gaussian
    density would suggest, because a wide range of z values map there.

    Why tanh squashing?
    - Many environments expect actions in some bounded range.
    - Clipping actions without correcting for the changed probabilities leads to
      an incorrect likelihood, which breaks policy gradient methods.

    Args:
        mean: Tensor of shape (batch_size, action_dim) - Gaussian mean
        log_std: Tensor of shape (action_dim,) or (batch_size, action_dim) - log std dev

    Returns:
        actions: Tensor of shape (batch_size, action_dim) - squashed actions in [-1, 1]
        log_probs: Tensor of shape (batch_size,) - log probabilities with Jacobian correction
    """
    # TODO: Implement this
    raise NotImplementedError("Implement sample_continuous_action")


def squashed_gaussian_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    squashed_action: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of a squashed (tanh-transformed) action.

    When evaluating the log probability of an action that was already squashed,
    we need to:
    1. Invert the tanh to get the original unbounded action: z = atanh(action)
    2. Compute the Gaussian log probability of z
    3. Apply the Jacobian correction for the tanh transform (see
       sample_continuous_action docstring for the derivation)

    This is used during PPO updates when we need to compute the log probability
    of actions that were sampled earlier (and stored as squashed actions).

    Args:
        mean: Tensor of shape (batch_size, action_dim) - Gaussian mean
        log_std: Tensor of shape (action_dim,) or (batch_size, action_dim)
        squashed_action: Tensor of shape (batch_size, action_dim) - actions in [-1, 1]

    Returns:
        log_probs: Tensor of shape (batch_size,) - log probabilities

    Nuances to handle:
        - Clamp squashed_action to (-0.999, 0.999) before atanh to avoid infinity
          at the boundaries (atanh(1) = inf, atanh(-1) = -inf)
        - The Jacobian correction term is: -sum(log(1 - action^2 + eps))

    """
    # TODO: Implement this
    raise NotImplementedError("Implement squashed_gaussian_log_prob")


def clip_action(
    action: torch.Tensor, low: float = -1.0, high: float = 1.0
) -> torch.Tensor:
    """
    Clip continuous actions to a specified range.

    While tanh naturally bounds actions to [-1, 1], explicit clipping is still
    useful for:
    1. Handling numerical edge cases
    2. Scaling to different action ranges
    3. Ensuring hard constraints are satisfied

    Args:
        action: Tensor of any shape containing continuous actions
        low: Minimum action value (default -1.0)
        high: Maximum action value (default 1.0)

    Returns:
        clipped_action: Tensor of same shape with values clipped to [low, high]
    """
    # TODO: Implement this
    raise NotImplementedError("Implement clip_action")


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

    PPO's key innovation is the clipped surrogate objective, which prevents
    the policy from changing too much in a single update.

    The probability ratio measures how much the policy has changed:
        ratio = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)

    The clipped objective is:
        L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    Why clipping?
    - If advantage > 0 (good action), we want to increase probability
      But we clip ratio at (1+eps) to prevent too large an increase
    - If advantage < 0 (bad action), we want to decrease probability
      But we clip ratio at (1-eps) to prevent too large a decrease

    We want to maximize this objective, but PyTorch optimizers minimize,
    so we return the negative of the mean objective.

    Args:
        log_probs: Tensor of shape (batch_size,) - log probs under current policy
        old_log_probs: Tensor of shape (batch_size,) - log probs under old policy
                       (from when we collected the data)
        advantages: Tensor of shape (batch_size,) - advantage estimates
        clip_epsilon: Clipping parameter (typically 0.2)

    Returns:
        loss: Scalar tensor. This is negative of the objective (for minimization).

    Nuances to handle:
        - The min() creates a "pessimistic bound" - we always take the worse estimate
          of the objective, whether the advantage is positive or negative.
        - With positive advantage: clipping prevents ratio from going too high
        - With negative advantage: clipping prevents ratio from going too low
        - Return the negative of the objective (we maximize objective but
          optimizers minimize loss).

    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_policy_loss")


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the value function loss (mean squared error).

    The value function V(s) estimates the expected return from state s.
    We train it to match the actual returns we observed.

    Loss = mean((V(s) - G)^2)

    where G is the actual return (computed from rewards).

    Args:
        values: Tensor of shape (batch_size,) containing value predictions V(s)
        returns: Tensor of shape (batch_size,) containing target returns G

    Returns:
        loss: Scalar tensor containing MSE loss
    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_value_loss")


def compute_entropy_bonus(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean entropy of a batch of categorical distributions.

    This is used as a bonus term in the PPO loss to encourage exploration:
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    Note the negative sign: we subtract the entropy loss, which means we're
    adding entropy to the objective (maximizing entropy = more exploration).

    Args:
        probs: Tensor of shape (batch_size, num_actions) containing action probabilities
               (i.e., after softmax). Unlike discrete_entropy which takes raw logits and
               applies softmax internally, this function expects pre-normalized probabilities
               that sum to 1 along the last dimension.

    Returns:
        entropy: Scalar tensor containing mean entropy across the batch

    Nuances to handle:
        - Add epsilon (1e-8) before taking log to handle the case where a probability
          is exactly 0. Without this, log(0) = -inf and you'll get NaN.
        - Return the mean entropy across the batch, not the sum.

    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_entropy_bonus")


# =============================================================================
# Part 4: Rollout Buffer for Vectorized Environments
# =============================================================================


class RolloutBuffer:
    """
    Buffer for storing trajectories collected from vectorized environments.

    In PPO, we collect a fixed number of steps from multiple parallel environments,
    then use all that data to update the policy. This buffer stores:
    - Observations (states)
    - Actions taken
    - Log probabilities of actions (under the old policy, for computing ratio)
    - Rewards received
    - Done flags (episode boundaries)
    - Value estimates (for computing advantages)

    Key concept: partial episodes
    When we collect num_steps from each environment, episodes may not be complete.
    For incomplete episodes, we need to "bootstrap" the value of the final state
    to estimate the return. This is handled in compute_returns_and_advantages.
    See the README for more details

    Example with num_steps=4 and num_envs=2:
        Env 0: [s0, s1, s2, s3] -> episode continues (need bootstrap)
        Env 1: [s0, s1, DONE, s0, s1] -> episode ended, new one started

    A useful thing to note: returns = advantages + values
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
            num_steps: Number of steps to collect per rollout (e.g., 128)
            num_envs: Number of parallel environments (e.g., 4)
            obs_shape: Shape of observations, e.g., (4,) for CartPole
            action_shape: Shape of actions. Use () for discrete actions (scalar),
                          (action_dim,) for continuous actions
            device: Device to store tensors on

        Storage needed: obs, actions, log_probs, rewards, dones, values.
        Think about what shape each tensor should be given num_steps, num_envs,
        and the observation/action shapes.
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # TODO: Initialize storage tensors
        raise NotImplementedError("Initialize RolloutBuffer storage")

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

        This is called once per environment step. Each input has data for
        all environments at the current timestep.

        Args:
            obs: shape (num_envs,) + obs_shape
            action: shape (num_envs,) + action_shape
            log_prob: shape (num_envs,)
            reward: shape (num_envs,)
            done: shape (num_envs,)
            value: shape (num_envs,)

        """
        # TODO: Implement this
        raise NotImplementedError("Implement RolloutBuffer.add")

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and GAE advantages for the collected rollout.

        This handles partial episodes correctly by bootstrapping with the
        value of the final state when the episode hasn't ended.

        Args:
            last_value: Value estimate for state after the last collected step.
                        Shape: (num_envs,). This is V(s_{num_steps}).
            last_done: Done flags for the last step. Shape: (num_envs,).
                       If last_done[i] = 1, env i's episode ended, so we don't bootstrap.
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            returns: shape (num_steps, num_envs) - target values for value function
            advantages: shape (num_steps, num_envs) - advantages for policy gradient

        Nuances to handle:
            - You need to build an extended values tensor of shape
              (num_steps + 1, num_envs) that includes the bootstrap value at the end,
              because compute_gae expects values to have one more element along dim 0.
            - When last_done=1 for an environment, the bootstrap value should be 0
              (the episode ended, so there's no future value to bootstrap from).

        """
        # TODO: Implement this
        raise NotImplementedError(
            "Implement RolloutBuffer.compute_returns_and_advantages"
        )

    def get_batches(
        self, batch_size: int, returns: torch.Tensor, advantages: torch.Tensor
    ):
        """
        Generate random minibatches for training.

        PPO trains on the collected data for multiple epochs, using random
        minibatches each time. This function:
        1. Flattens the (num_steps, num_envs) data into (total_samples,)
        2. Shuffles the indices
        3. Yields batches of the specified size

        Args:
            batch_size: Number of samples per minibatch
            returns: shape (num_steps, num_envs)
            advantages: shape (num_steps, num_envs)

        Yields:
            dict with keys: obs, actions, log_probs, returns, advantages, values
            Each value has shape (batch_size, ...)

        Nuances to handle:
            - Flatten the (num_steps, num_envs) dimensions into a single dimension
              before creating batches.
            - Shuffle the indices to ensure random minibatches.
            - Preserve the trailing dimensions (obs_shape, action_shape) when reshaping.
        """
        # TODO: Implement this
        raise NotImplementedError("Implement RolloutBuffer.get_batches")
