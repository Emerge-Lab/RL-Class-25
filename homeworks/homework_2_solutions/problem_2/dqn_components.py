"""
DQN Components - Homework 2, Problem 2 (Solutions)

This file contains the core building blocks of Deep Q-Network (DQN):
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

import random
from collections import deque
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.buffer = deque(maxlen=capacity)

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
        self.buffer.append(Transition(state, action, reward, next_state, done))

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
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)


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
        self.buffer = ReplayBuffer(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

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
        """
        self.n_step_buffer.append(Transition(state, action, reward, next_state, done))

        if done:
            # Flush all partial n-step transitions on episode boundary
            while len(self.n_step_buffer) > 0:
                n_step_return, nth_state, nth_done = self._compute_nstep()
                first = self.n_step_buffer.popleft()
                self.buffer.push(
                    first.state, first.action, n_step_return, nth_state, nth_done
                )
        elif len(self.n_step_buffer) == self.n_step:
            # Full n-step transition ready
            n_step_return, nth_state, nth_done = self._compute_nstep()
            first = self.n_step_buffer.popleft()
            self.buffer.push(
                first.state, first.action, n_step_return, nth_state, nth_done
            )

    def _compute_nstep(self):
        """Compute n-step discounted return from the n-step buffer."""
        n_step_return = 0.0
        for i, t in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma**i) * t.reward
            if t.done:
                return n_step_return, t.next_state, True
        last = self.n_step_buffer[-1]
        return n_step_return, last.next_state, last.done

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
    """
    states = torch.tensor(
        np.array([t.state for t in batch]), dtype=torch.float32, device=device
    )
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor(
        [t.reward for t in batch], dtype=torch.float32, device=device
    )
    next_states = torch.tensor(
        np.array([t.next_state for t in batch]), dtype=torch.float32, device=device
    )
    dones = torch.tensor(
        [float(t.done) for t in batch], dtype=torch.float32, device=device
    )

    return states, actions, rewards, next_states, dones


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
        - For ties in Q-values, argmax returns the first occurrence
    """
    # Flatten if needed
    if q_values.dim() > 1:
        q_values = q_values.squeeze(0)

    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        return q_values.argmax().item()


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
    """
    fraction = min(step / decay_steps, 1.0)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


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

    Args:
        rewards: Reward tensor of shape (batch_size,)
        next_q_values: Q-values for next states, shape (batch_size, num_actions)
        dones: Done flags tensor of shape (batch_size,)
        gamma: Discount factor

    Returns:
        td_targets: TD target tensor of shape (batch_size,)
    """
    # Max Q-value for each next state
    max_next_q = next_q_values.max(dim=1).values

    # TD target: r + gamma * max Q(s', a') * (1 - done)
    td_targets = rewards + gamma * max_next_q * (1 - dones)

    return td_targets.detach()


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

    This reduces overestimation bias in standard DQN.

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
        - No gradients should flow through target computation
    """
    with torch.no_grad():
        # Use online network to select best actions
        online_q_next = online_network(next_states)
        best_actions = online_q_next.argmax(dim=1, keepdim=True)

        # Use target network to evaluate those actions
        target_q_next = target_network(next_states)
        next_q_values = target_q_next.gather(1, best_actions).squeeze(1)

        # TD target
        td_targets = rewards + gamma * next_q_values * (1 - dones)

    return td_targets


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

    Args:
        q_values: Q-values from online network, shape (batch_size, num_actions)
        actions: Actions taken, shape (batch_size,)
        td_targets: TD targets, shape (batch_size,)
        loss_type: "huber" (smooth L1) or "mse"

    Returns:
        loss: Scalar loss tensor

    Nuances to handle:
        - Use gather to select Q-values for the actions that were taken
        - Huber loss is more robust to outliers than MSE
        - TD targets should be detached (no gradient flow through targets)
    """
    # Select Q-values for actions taken
    # q_values: (batch_size, num_actions) -> (batch_size, 1) -> (batch_size,)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute loss
    if loss_type == "huber":
        loss = F.smooth_l1_loss(q_selected, td_targets.detach())
    elif loss_type == "mse":
        loss = F.mse_loss(q_selected, td_targets.detach())
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


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

    This is also known as Polyak averaging.

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
    with torch.no_grad():
        for online_param, target_param in zip(
            online_network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )


def hard_update(online_network: nn.Module, target_network: nn.Module) -> None:
    """
    Hard update of target network parameters.

    Copies all parameters from online network to target network.
    Equivalent to soft_update with tau=1.

    Args:
        online_network: The Q-network being trained
        target_network: The target Q-network to update
    """
    target_network.load_state_dict(online_network.state_dict())


# =============================================================================
# Part 6: Q-Network Architecture
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
