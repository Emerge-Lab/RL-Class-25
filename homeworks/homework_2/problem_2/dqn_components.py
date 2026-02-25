"""
DQN Components - Homework 2, Problem 2

This file contains the core building blocks of Deep Q-Network (DQN).
Your task is to implement each function marked with TODO.

Components to implement:
1. Replay Buffer - Store and sample experience
2. Epsilon-greedy exploration
3. TD target computation (standard and Double DQN)
4. TD loss computation
5. Target network updates (soft and hard)

Based on the original DQN paper and Double DQN:
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
"""

from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Part 1: Replay Buffer
# =============================================================================


class Transition(NamedTuple):
    """A single transition from the environment."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions (s, a, r, s', done) and samples random minibatches.
    This breaks the correlation between consecutive samples and stabilizes training.

    Key properties:
    - Fixed capacity with FIFO eviction (oldest transitions removed first)
    - Uniform random sampling for minibatches
    - Efficient storage using a circular buffer (deque)
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        # TODO: Implement this
        raise NotImplementedError("Implement ReplayBuffer.__init__")

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended

        """
        # TODO: Implement this
        raise NotImplementedError("Implement ReplayBuffer.push")

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of Transition namedtuples

        Nuances to handle:
            - Sample without replacement (each transition appears at most once)
            - If batch_size > len(buffer), this will raise an error (expected)
        """
        # TODO: Implement this
        raise NotImplementedError("Implement ReplayBuffer.sample")

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        # TODO: Implement this
        raise NotImplementedError("Implement ReplayBuffer.__len__")


class NStepReplayBuffer:
    """
    N-step replay buffer for DQN.

    Instead of storing single-step transitions (s, a, r, s', done), this buffer
    accumulates n consecutive transitions and stores the n-step return:

        R_n = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^{n-1} * r_{n-1}

    The stored transition becomes (s_0, a_0, R_n, s_n, done_n), where s_n is the
    state n steps later and done_n indicates if any episode boundary was hit.

    When computing TD targets with n-step returns, use gamma^n instead of gamma:
        y = R_n + gamma^n * Q(s_n, a') * (1 - done_n)

    This can improve credit assignment by propagating reward information
    n steps backward in a single update. We don't need it for the Cartpole
    problem we're going to solve here but I wanted you to be aware of the technique.
    """

    def __init__(self, capacity: int, n_step: int, gamma: float):
        """
        Initialize the n-step replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns
            gamma: Discount factor for computing n-step returns
        """
        # TODO: Implement this
        raise NotImplementedError("Implement NStepReplayBuffer.__init__")

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition. When n transitions accumulate (or episode ends),
        compute the n-step return and push to the underlying replay buffer.

        Logic:
            1. Append the transition to the n-step buffer.
            2. If done: flush all remaining transitions in the n-step buffer.
               For each, compute the n-step return from the remaining transitions,
               pop the oldest, and push (oldest.state, oldest.action, R_n, nth_state, nth_done)
               to the main buffer.
            3. Else if the n-step buffer is full (length == n_step):
               Compute the n-step return, pop the oldest transition, and push to main buffer.
        """
        # TODO: Implement this
        raise NotImplementedError("Implement NStepReplayBuffer.push")

    def _compute_nstep(self):
        """
        Compute n-step discounted return from the n-step buffer.

        Returns:
            n_step_return: The discounted sum of rewards
            nth_state: The state at the end of the n-step sequence
            nth_done: Whether the episode ended within the sequence
        """
        # TODO: Implement this
        raise NotImplementedError("Implement NStepReplayBuffer._compute_nstep")

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return self.buffer.sample(batch_size)

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)


def batch_to_tensors(
    batch: List[Transition], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of transitions to tensors.

    Args:
        batch: List of Transition namedtuples
        device: Device to create tensors on

    Returns:
        states: FloatTensor of shape (batch_size, state_dim)
        actions: LongTensor of shape (batch_size,)
        rewards: FloatTensor of shape (batch_size,)
        next_states: FloatTensor of shape (batch_size, state_dim)
        dones: FloatTensor of shape (batch_size,)

    Nuances to handle:
        - Actions must be LongTensor for use with gather()
        - Dones should be float (0.0 or 1.0) for easy multiplication
    """
    # TODO: Implement this
    raise NotImplementedError("Implement batch_to_tensors")


# =============================================================================
# Part 2: Epsilon-Greedy Exploration
# =============================================================================


def epsilon_greedy_action(
    q_values: torch.Tensor, epsilon: float, num_actions: int
) -> int:
    """
    Select an action using epsilon-greedy exploration.

    With probability epsilon, select a random action.
    With probability (1 - epsilon), select the greedy action (highest Q-value).

    Args:
        q_values: Q-values for each action, shape (num_actions,) or (1, num_actions)
        epsilon: Exploration probability (0 = greedy, 1 = random)
        num_actions: Number of possible actions

    Returns:
        action: int — selected action index

    Nuances to handle:
        - q_values may be 1D or 2D (with batch dim of 1) - squeeze if needed
        - When epsilon=0, always return greedy action
        - When epsilon=1, always return random action
        - For ties in Q-values, argmax returns the first occurrence
    """
    # TODO: Implement this
    raise NotImplementedError("Implement epsilon_greedy_action")


def linear_epsilon_decay(
    step: int, epsilon_start: float, epsilon_end: float, decay_steps: int
) -> float:
    """
    Compute epsilon using linear decay schedule.

    Epsilon decreases linearly from epsilon_start to epsilon_end
    over decay_steps steps, then stays at epsilon_end.

    Args:
        step: Current training step
        epsilon_start: Initial epsilon value
        epsilon_end: Final epsilon value
        decay_steps: Number of steps for decay

    Returns:
        epsilon: float — current epsilon value

    Nuances to handle:
        - After decay_steps, epsilon should stay at epsilon_end (not go below)
        - At step=0, epsilon should be epsilon_start
        - At step=decay_steps, epsilon should be epsilon_end
    """
    # TODO: Implement this
    raise NotImplementedError("Implement linear_epsilon_decay")


# =============================================================================
# Part 3: TD Target Computation
# =============================================================================


def compute_td_target(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Compute TD targets for standard DQN.

    TD target = r + gamma * max_a' Q(s', a') * (1 - done)

    The key insight: we want to bootstrap from the maximum Q-value of the
    next state, but only if the episode hasn't ended.

    Args:
        rewards: Reward tensor of shape (batch_size,)
        next_q_values: Q-values for next states, shape (batch_size, num_actions)
        dones: Done flags tensor of shape (batch_size,)
        gamma: Discount factor

    Returns:
        td_targets: TD target tensor of shape (batch_size,)

    Nuances to handle:
        - When done=1, the target should just be the reward (no future value)
        - The (1-done) term "masks out" the Q-value for terminal states
        - Return detached targets to prevent gradients flowing through the target
          computation (consistent with compute_double_dqn_target which uses torch.no_grad())
    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_td_target")


def compute_double_dqn_target(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    online_network: nn.Module,
    target_network: nn.Module,
) -> torch.Tensor:
    """
    Compute TD targets for Double DQN.

    Double DQN decouples action selection from action evaluation:
    - Use online network to select the best action: a* = argmax_a Q_online(s', a)
    - Use target network to evaluate that action: Q_target(s', a*)

    This reduces the overestimation bias present in standard DQN, where using
    max causes a positive bias (max of noisy estimates is biased upward).

    TD target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)

    Args:
        rewards: Reward tensor of shape (batch_size,)
        next_states: Next state tensor of shape (batch_size, state_dim)
        dones: Done flags tensor of shape (batch_size,)
        gamma: Discount factor
        online_network: The Q-network being trained
        target_network: The target Q-network (delayed copy)

    Returns:
        td_targets: TD target tensor of shape (batch_size,)

    Nuances to handle:
        - Use online network for action selection (argmax)
        - Use target network for value evaluation
        - No gradients should flow through target computation (use torch.no_grad())
        - Use gather() to select Q-values for the chosen actions
    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_double_dqn_target")


# =============================================================================
# Part 4: TD Loss Computation
# =============================================================================


def compute_td_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    loss_type: str = "huber",
) -> torch.Tensor:
    """
    Compute the TD loss for DQN.

    The loss measures how far our Q-value predictions are from the TD targets.
    We only compute loss for the actions that were actually taken.

    Args:
        q_values: Q-values from online network, shape (batch_size, num_actions)
        actions: Actions taken, shape (batch_size,)
        td_targets: TD targets, shape (batch_size,)
        loss_type: "huber" (smooth L1) or "mse"

    Returns:
        loss: Scalar loss tensor

    Nuances to handle:
        - Use gather to select Q-values for the actions that were taken
        - Huber loss is more robust to outliers than MSE (default choice)
        - TD targets should be detached (no gradient flow through targets)
    """
    # TODO: Implement this
    raise NotImplementedError("Implement compute_td_loss")


# =============================================================================
# Part 5: Target Network Updates
# =============================================================================


def soft_update(
    online_network: nn.Module, target_network: nn.Module, tau: float
) -> None:
    """
    Soft update of target network parameters.

    For each parameter:
        θ_target = τ * θ_online + (1 - τ) * θ_target

    This is also known as Polyak averaging. It provides a smoother
    update than hard copying, which can help stability.

    Args:
        online_network: The Q-network being trained
        target_network: The target Q-network to update
        tau: Interpolation parameter (0 < tau <= 1)
             tau=1 means hard update (copy)
             tau=0.005 is a common choice for soft updates

    Nuances to handle:
        - Update should be done in-place on target network parameters
        - No gradients should be computed for this operation
    """
    # TODO: Implement this
    raise NotImplementedError("Implement soft_update")


def hard_update(online_network: nn.Module, target_network: nn.Module) -> None:
    """
    Hard update of target network parameters.

    Copies all parameters from online network to target network.
    Equivalent to soft_update with tau=1.

    Args:
        online_network: The Q-network being trained
        target_network: The target Q-network to update
    """
    # TODO: Implement this
    raise NotImplementedError("Implement hard_update")


# =============================================================================
# Part 6: Q-Network Architecture (PROVIDED - No implementation needed)
# =============================================================================


class QNetwork(nn.Module):
    """
    Simple fully-connected Q-network for discrete action spaces.

    Takes a state as input and outputs Q-values for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            q_values: Q-values of shape (batch_size, action_dim)
        """
        return self.network(state)
