"""
DQN Components Test Suite - Homework 2, Problem 2

Tests for the DQN building blocks implemented in dqn_components.py.

Run with: uv run python test_components.py
"""

import numpy as np
import torch
import torch.nn as nn

from dqn_components import (
    ReplayBuffer,
    NStepReplayBuffer,
    Transition,
    batch_to_tensors,
    epsilon_greedy_action,
    linear_epsilon_decay,
    compute_td_target,
    compute_double_dqn_target,
    compute_td_loss,
    soft_update,
    hard_update,
    QNetwork,
)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


def test_passed(name: str):
    print(f"  [{Colors.GREEN}PASS{Colors.RESET}] {name}")


def test_failed(name: str, msg: str):
    print(f"  [{Colors.RED}FAIL{Colors.RESET}] {name}")
    print(f"         {msg}")


def assert_close(actual, expected, name: str, rtol: float = 1e-4, atol: float = 1e-6):
    """Assert two tensors/values are close."""
    if isinstance(actual, torch.Tensor):
        actual = actual.detach()
    if isinstance(expected, torch.Tensor):
        expected = expected.detach()

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            raise AssertionError(
                f"Tensors not close.\nExpected:\n{expected}\nActual:\n{actual}"
            )
    else:
        if abs(actual - expected) > atol + rtol * abs(expected):
            raise AssertionError(f"Values not close. Expected {expected}, got {actual}")


# =============================================================================
# Part 1: Replay Buffer Tests
# =============================================================================


def test_replay_buffer():
    """Test ReplayBuffer basic operations."""
    try:
        buffer = ReplayBuffer(capacity=100)

        # Test empty buffer
        assert len(buffer) == 0, "Empty buffer should have length 0"

        # Add some transitions
        for i in range(10):
            state = np.array([i, i + 1], dtype=np.float32)
            buffer.push(
                state=state,
                action=i % 2,
                reward=float(i),
                next_state=state + 1,
                done=(i == 9),
            )

        assert len(buffer) == 10, f"Buffer should have 10 items, got {len(buffer)}"

        # Test sampling
        batch = buffer.sample(5)
        assert len(batch) == 5, f"Batch should have 5 items, got {len(batch)}"
        assert all(
            isinstance(t, Transition) for t in batch
        ), "Batch items should be Transitions"

        test_passed("replay_buffer")
        return True

    except Exception as e:
        test_failed("replay_buffer", str(e))
        return False


def test_replay_buffer_capacity():
    """Test that replay buffer respects capacity limit."""
    try:
        buffer = ReplayBuffer(capacity=5)

        # Add 10 transitions
        for i in range(10):
            state = np.array([i], dtype=np.float32)
            buffer.push(state, 0, 0.0, state, False)

        # Buffer should only have 5 items
        assert len(buffer) == 5, f"Buffer should have 5 items, got {len(buffer)}"

        # The oldest transitions should be gone (FIFO)
        # Check that the states in buffer are from the last 5 pushes
        states_in_buffer = [t.state[0] for t in buffer.buffer]
        assert min(states_in_buffer) >= 5, "Oldest transitions should be evicted"

        test_passed("replay_buffer_capacity")
        return True

    except Exception as e:
        test_failed("replay_buffer_capacity", str(e))
        return False


def test_batch_to_tensors():
    """Test batch_to_tensors conversion."""
    try:
        batch = [
            Transition(np.array([1.0, 2.0]), 0, 1.0, np.array([2.0, 3.0]), False),
            Transition(np.array([3.0, 4.0]), 1, -1.0, np.array([4.0, 5.0]), True),
        ]

        device = torch.device("cpu")
        states, actions, rewards, next_states, dones = batch_to_tensors(batch, device)

        # Check shapes
        assert states.shape == (
            2,
            2,
        ), f"states shape should be (2, 2), got {states.shape}"
        assert actions.shape == (
            2,
        ), f"actions shape should be (2,), got {actions.shape}"
        assert rewards.shape == (
            2,
        ), f"rewards shape should be (2,), got {rewards.shape}"
        assert next_states.shape == (2, 2), "next_states shape should be (2, 2)"
        assert dones.shape == (2,), f"dones shape should be (2,), got {dones.shape}"

        # Check dtypes
        assert (
            actions.dtype == torch.long
        ), f"actions should be long, got {actions.dtype}"
        assert (
            dones.dtype == torch.float32
        ), f"dones should be float32, got {dones.dtype}"

        # Check values
        assert_close(rewards, torch.tensor([1.0, -1.0]), "rewards")
        assert_close(dones, torch.tensor([0.0, 1.0]), "dones")

        test_passed("batch_to_tensors")
        return True

    except Exception as e:
        test_failed("batch_to_tensors", str(e))
        return False


def test_nstep_buffer_basic():
    """Test NStepReplayBuffer computes n-step returns correctly."""
    try:
        # 3-step buffer with gamma=0.9
        buffer = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.9)

        # Push 5 transitions (no episode boundary)
        for i in range(5):
            state = np.array([float(i)], dtype=np.float32)
            buffer.push(state, 0, 1.0, state + 1, False)

        # After 5 pushes with n=3, we should have 3 transitions in main buffer
        # (transitions 0,1,2 are ready; 3,4 still in n-step buffer)
        assert len(buffer) == 3, f"Expected 3 transitions, got {len(buffer)}"

        # Check the first transition's n-step return
        # R_3 = 1.0 + 0.9*1.0 + 0.81*1.0 = 2.71
        t = buffer.buffer.buffer[0]
        expected_return = 1.0 + 0.9 * 1.0 + 0.81 * 1.0
        assert (
            abs(t.reward - expected_return) < 1e-5
        ), f"Expected n-step return {expected_return}, got {t.reward}"

        # The next_state should be state 3 (3 steps ahead from state 0)
        assert (
            abs(t.next_state[0] - 3.0) < 1e-5
        ), f"Expected next_state=3.0, got {t.next_state[0]}"

        test_passed("nstep_buffer_basic")
        return True

    except Exception as e:
        test_failed("nstep_buffer_basic", str(e))
        return False


def test_nstep_buffer_episode_boundary():
    """Test NStepReplayBuffer handles episode boundaries correctly."""
    try:
        buffer = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.9)

        # Push transitions with episode ending at step 1
        buffer.push(np.array([0.0]), 0, 1.0, np.array([1.0]), False)
        buffer.push(np.array([1.0]), 0, 2.0, np.array([2.0]), True)  # done!

        # On done, all pending transitions should be flushed
        assert len(buffer) == 2, f"Expected 2 transitions, got {len(buffer)}"

        # First transition: R = 1.0 + 0.9*2.0 = 2.8 (2 steps, episode ends)
        t0 = buffer.buffer.buffer[0]
        expected = 1.0 + 0.9 * 2.0
        assert (
            abs(t0.reward - expected) < 1e-5
        ), f"Expected return {expected}, got {t0.reward}"
        assert t0.done is True, "Should be done (episode ended within n steps)"

        # Second transition: R = 2.0 (just the terminal reward)
        t1 = buffer.buffer.buffer[1]
        assert abs(t1.reward - 2.0) < 1e-5, f"Expected return 2.0, got {t1.reward}"
        assert t1.done is True, "Terminal transition should have done=True"

        test_passed("nstep_buffer_episode_boundary")
        return True

    except Exception as e:
        test_failed("nstep_buffer_episode_boundary", str(e))
        return False


def test_nstep_buffer_one_step():
    """Test NStepReplayBuffer with n_step=1 behaves like regular ReplayBuffer."""
    try:
        buffer = NStepReplayBuffer(capacity=100, n_step=1, gamma=0.99)

        for i in range(5):
            state = np.array([float(i)], dtype=np.float32)
            buffer.push(state, 0, float(i), state + 1, False)

        # With n=1, every push should immediately go to main buffer
        assert len(buffer) == 5, f"Expected 5 transitions, got {len(buffer)}"

        # Returns should just be the raw rewards (no discounting across steps)
        for i in range(5):
            t = buffer.buffer.buffer[i]
            assert (
                abs(t.reward - float(i)) < 1e-5
            ), f"Expected reward {float(i)}, got {t.reward}"

        test_passed("nstep_buffer_one_step")
        return True

    except Exception as e:
        test_failed("nstep_buffer_one_step", str(e))
        return False


# =============================================================================
# Part 2: Epsilon-Greedy Tests
# =============================================================================


def test_epsilon_greedy_greedy():
    """Test epsilon_greedy_action with epsilon=0 (pure greedy)."""
    try:
        q_values = torch.tensor([1.0, 5.0, 3.0])

        # With epsilon=0, should always pick action 1 (highest Q-value)
        for _ in range(10):
            action = epsilon_greedy_action(q_values, epsilon=0.0, num_actions=3)
            assert action == 1, f"With epsilon=0, should pick action 1, got {action}"

        test_passed("epsilon_greedy_greedy")
        return True

    except Exception as e:
        test_failed("epsilon_greedy_greedy", str(e))
        return False


def test_epsilon_greedy_random():
    """Test epsilon_greedy_action with epsilon=1 (pure random)."""
    try:
        q_values = torch.tensor([100.0, 0.0, 0.0])  # Strong preference for action 0

        # With epsilon=1, should explore all actions
        actions = [
            epsilon_greedy_action(q_values, epsilon=1.0, num_actions=3)
            for _ in range(100)
        ]

        # Check that we see at least 2 different actions (very likely with 100 samples)
        unique_actions = set(actions)
        assert (
            len(unique_actions) >= 2
        ), f"With epsilon=1, should see multiple actions, got {unique_actions}"

        test_passed("epsilon_greedy_random")
        return True

    except Exception as e:
        test_failed("epsilon_greedy_random", str(e))
        return False


def test_epsilon_greedy_2d_input():
    """Test epsilon_greedy_action handles 2D input (batch dim)."""
    try:
        q_values = torch.tensor([[1.0, 5.0, 3.0]])  # Shape (1, 3)

        action = epsilon_greedy_action(q_values, epsilon=0.0, num_actions=3)
        assert action == 1, f"Should handle 2D input, expected 1, got {action}"

        test_passed("epsilon_greedy_2d_input")
        return True

    except Exception as e:
        test_failed("epsilon_greedy_2d_input", str(e))
        return False


def test_linear_epsilon_decay():
    """Test linear_epsilon_decay schedule."""
    try:
        # At step 0, should be epsilon_start
        eps = linear_epsilon_decay(
            step=0, epsilon_start=1.0, epsilon_end=0.1, decay_steps=100
        )
        assert_close(eps, 1.0, "epsilon at step 0")

        # At step 50, should be midway
        eps = linear_epsilon_decay(
            step=50, epsilon_start=1.0, epsilon_end=0.1, decay_steps=100
        )
        assert_close(eps, 0.55, "epsilon at step 50")

        # At step 100, should be epsilon_end
        eps = linear_epsilon_decay(
            step=100, epsilon_start=1.0, epsilon_end=0.1, decay_steps=100
        )
        assert_close(eps, 0.1, "epsilon at step 100")

        # After decay_steps, should stay at epsilon_end
        eps = linear_epsilon_decay(
            step=200, epsilon_start=1.0, epsilon_end=0.1, decay_steps=100
        )
        assert_close(eps, 0.1, "epsilon after decay")

        test_passed("linear_epsilon_decay")
        return True

    except Exception as e:
        test_failed("linear_epsilon_decay", str(e))
        return False


# =============================================================================
# Part 3: TD Target Tests
# =============================================================================


def test_compute_td_target_basic():
    """Test compute_td_target for non-terminal states."""
    try:
        rewards = torch.tensor([1.0, 2.0])
        next_q_values = torch.tensor(
            [
                [1.0, 5.0, 3.0],  # max = 5
                [2.0, 1.0, 4.0],  # max = 4
            ]
        )
        dones = torch.tensor([0.0, 0.0])
        gamma = 0.99

        targets = compute_td_target(rewards, next_q_values, dones, gamma)

        # target = r + gamma * max_q
        expected = torch.tensor([1.0 + 0.99 * 5.0, 2.0 + 0.99 * 4.0])
        assert_close(targets, expected, "td_targets for non-terminal")

        test_passed("compute_td_target_basic")
        return True

    except Exception as e:
        test_failed("compute_td_target_basic", str(e))
        return False


def test_compute_td_target_terminal():
    """Test compute_td_target for terminal states."""
    try:
        rewards = torch.tensor([1.0, 10.0])
        next_q_values = torch.tensor(
            [
                [1.0, 5.0, 3.0],
                [100.0, 200.0, 300.0],  # These should be ignored for terminal
            ]
        )
        dones = torch.tensor([0.0, 1.0])  # Second transition is terminal
        gamma = 0.99

        targets = compute_td_target(rewards, next_q_values, dones, gamma)

        # For terminal state, target = reward only
        expected = torch.tensor([1.0 + 0.99 * 5.0, 10.0])
        assert_close(targets, expected, "td_targets with terminal")

        test_passed("compute_td_target_terminal")
        return True

    except Exception as e:
        test_failed("compute_td_target_terminal", str(e))
        return False


def test_compute_double_dqn_target():
    """Test compute_double_dqn_target."""
    try:
        # Create simple networks with known outputs
        class FixedQNetwork(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = values

            def forward(self, x):
                batch_size = x.shape[0]
                return self.values.expand(batch_size, -1)

        # Online network: action 0 is best (10 > 5 > 1)
        online_net = FixedQNetwork(torch.tensor([[10.0, 5.0, 1.0]]))
        # Target network: different values (1 for action 0)
        target_net = FixedQNetwork(torch.tensor([[1.0, 100.0, 50.0]]))

        rewards = torch.tensor([1.0])
        next_states = torch.tensor([[0.0, 0.0]])  # Dummy states
        dones = torch.tensor([0.0])
        gamma = 0.99

        targets = compute_double_dqn_target(
            rewards, next_states, dones, gamma, online_net, target_net
        )

        # Double DQN: online selects action 0, target evaluates action 0
        # target = 1.0 + 0.99 * 1.0 = 1.99
        expected = torch.tensor([1.99])
        assert_close(targets, expected, "double_dqn_target")

        test_passed("compute_double_dqn_target")
        return True

    except Exception as e:
        test_failed("compute_double_dqn_target", str(e))
        return False


def test_compute_double_dqn_target_terminal():
    """Test compute_double_dqn_target with terminal states (done=1)."""
    try:

        class FixedQNetwork(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = values

            def forward(self, x):
                batch_size = x.shape[0]
                return self.values.expand(batch_size, -1)

        online_net = FixedQNetwork(torch.tensor([[10.0, 5.0, 1.0]]))
        target_net = FixedQNetwork(torch.tensor([[1.0, 100.0, 50.0]]))

        rewards = torch.tensor([1.0, 5.0])
        next_states = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        dones = torch.tensor([0.0, 1.0])  # Second state is terminal
        gamma = 0.99

        targets = compute_double_dqn_target(
            rewards, next_states, dones, gamma, online_net, target_net
        )

        # Non-terminal: r + gamma * Q_target(s', a*) = 1.0 + 0.99 * 1.0 = 1.99
        # Terminal: just reward = 5.0 (Q-values should be masked out)
        expected = torch.tensor([1.99, 5.0])
        assert_close(targets, expected, "double_dqn_target_terminal")

        test_passed("compute_double_dqn_target_terminal")
        return True

    except Exception as e:
        test_failed("compute_double_dqn_target_terminal", str(e))
        return False


def test_compute_td_target_detach():
    """Test that compute_td_target returns detached targets (no gradient flow)."""
    try:
        rewards = torch.tensor([1.0])
        next_q_values = torch.tensor([[1.0, 5.0, 3.0]], requires_grad=True)
        dones = torch.tensor([0.0])
        gamma = 0.99

        targets = compute_td_target(rewards, next_q_values, dones, gamma)

        # Targets should be detached — no gradient flows back through them
        assert (
            not targets.requires_grad
        ), "TD targets should be detached (requires_grad should be False)"

        test_passed("compute_td_target_detach")
        return True

    except Exception as e:
        test_failed("compute_td_target_detach", str(e))
        return False


def test_compute_double_dqn_target_detach():
    """Test that compute_double_dqn_target returns detached targets."""
    try:

        class FixedQNetwork(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.linear = nn.Linear(2, 3)  # Need real params for grad check
                self.values = values

            def forward(self, x):
                batch_size = x.shape[0]
                return self.values.expand(batch_size, -1)

        online_net = FixedQNetwork(torch.tensor([[10.0, 5.0, 1.0]]))
        target_net = FixedQNetwork(torch.tensor([[1.0, 100.0, 50.0]]))

        rewards = torch.tensor([1.0])
        next_states = torch.tensor([[0.0, 0.0]])
        dones = torch.tensor([0.0])
        gamma = 0.99

        targets = compute_double_dqn_target(
            rewards, next_states, dones, gamma, online_net, target_net
        )

        # Targets should be detached — no gradient flows through the target computation
        assert (
            not targets.requires_grad
        ), "Double DQN targets should be detached (requires_grad should be False)"

        test_passed("compute_double_dqn_target_detach")
        return True

    except Exception as e:
        test_failed("compute_double_dqn_target_detach", str(e))
        return False


# =============================================================================
# Part 4: TD Loss Tests
# =============================================================================


def test_compute_td_loss_huber():
    """Test compute_td_loss with Huber loss."""
    try:
        q_values = torch.tensor(
            [
                [1.0, 5.0, 3.0],
                [2.0, 1.0, 4.0],
            ]
        )
        actions = torch.tensor([1, 2])  # Selected actions
        td_targets = torch.tensor([5.0, 4.0])  # Targets match selected Q-values

        loss = compute_td_loss(q_values, actions, td_targets, loss_type="huber")

        # When predictions match targets, loss should be ~0
        assert loss.item() < 1e-6, f"Loss should be ~0, got {loss.item()}"

        test_passed("compute_td_loss_huber")
        return True

    except Exception as e:
        test_failed("compute_td_loss_huber", str(e))
        return False


def test_compute_td_loss_nonzero():
    """Test compute_td_loss with actual error."""
    try:
        q_values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
            ]
        )
        actions = torch.tensor([1])  # Q[1] = 2.0
        td_targets = torch.tensor([4.0])  # Target = 4.0, error = 2.0

        loss_huber = compute_td_loss(q_values, actions, td_targets, loss_type="huber")
        loss_mse = compute_td_loss(q_values, actions, td_targets, loss_type="mse")

        # MSE loss = (2-4)^2 = 4
        assert_close(loss_mse, torch.tensor(4.0), "mse_loss")

        # Huber loss with delta=1: for |error|=2 > 1, loss = |error| - 0.5 = 1.5
        assert_close(loss_huber, torch.tensor(1.5), "huber_loss")

        test_passed("compute_td_loss_nonzero")
        return True

    except Exception as e:
        test_failed("compute_td_loss_nonzero", str(e))
        return False


def test_compute_td_loss_gradient():
    """Test that TD loss computes gradients correctly."""
    try:
        q_values = torch.tensor([[1.0, 2.0]], requires_grad=True)
        actions = torch.tensor([0])
        td_targets = torch.tensor([3.0])

        loss = compute_td_loss(q_values, actions, td_targets)
        loss.backward()

        assert q_values.grad is not None, "Gradient should exist"
        # Gradient should be non-zero for action 0, zero for action 1
        assert (
            q_values.grad[0, 0] != 0
        ), "Gradient for selected action should be non-zero"
        assert (
            q_values.grad[0, 1] == 0
        ), "Gradient for non-selected action should be zero"

        test_passed("compute_td_loss_gradient")
        return True

    except Exception as e:
        test_failed("compute_td_loss_gradient", str(e))
        return False


def test_compute_td_loss_detach():
    """Test that TD targets are detached (no gradient flows through targets)."""
    try:
        q_values = torch.tensor([[1.0, 2.0]], requires_grad=True)
        actions = torch.tensor([0])

        # Create td_targets with requires_grad=True
        # compute_td_loss should detach them internally
        td_targets = torch.tensor([3.0], requires_grad=True)

        loss = compute_td_loss(q_values, actions, td_targets)
        loss.backward()

        # td_targets should NOT have gradients (detached inside compute_td_loss)
        assert (
            td_targets.grad is None
        ), "td_targets should be detached inside compute_td_loss (no gradient should flow through targets)"

        # q_values SHOULD have gradients
        assert q_values.grad is not None, "q_values should have gradients"

        test_passed("compute_td_loss_detach")
        return True

    except Exception as e:
        test_failed("compute_td_loss_detach", str(e))
        return False


# =============================================================================
# Part 5: Target Network Update Tests
# =============================================================================


def test_soft_update():
    """Test soft_update with tau=0.1."""
    try:
        # Create simple networks
        online = nn.Linear(2, 2, bias=False)
        target = nn.Linear(2, 2, bias=False)

        # Set known weights
        with torch.no_grad():
            online.weight.fill_(1.0)
            target.weight.fill_(0.0)

        # Soft update with tau=0.1
        soft_update(online, target, tau=0.1)

        # target = 0.1 * 1.0 + 0.9 * 0.0 = 0.1
        expected = torch.full_like(target.weight, 0.1)
        assert_close(target.weight, expected, "soft_update tau=0.1")

        test_passed("soft_update")
        return True

    except Exception as e:
        test_failed("soft_update", str(e))
        return False


def test_soft_update_tau_one():
    """Test that soft_update with tau=1 is equivalent to hard_update."""
    try:
        online = nn.Linear(2, 2, bias=False)
        target = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            online.weight.fill_(5.0)
            target.weight.fill_(0.0)

        soft_update(online, target, tau=1.0)

        # With tau=1, target should equal online
        assert_close(target.weight, online.weight, "soft_update tau=1")

        test_passed("soft_update_tau_one")
        return True

    except Exception as e:
        test_failed("soft_update_tau_one", str(e))
        return False


def test_hard_update():
    """Test hard_update copies all weights."""
    try:
        online = QNetwork(4, 2, hidden_dim=8)
        target = QNetwork(4, 2, hidden_dim=8)

        # Hard update
        hard_update(online, target)

        # Now they should be equal
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            assert torch.equal(
                p_online, p_target
            ), "Weights should match after hard_update"

        test_passed("hard_update")
        return True

    except Exception as e:
        test_failed("hard_update", str(e))
        return False


# =============================================================================
# Part 6: Q-Network Tests
# =============================================================================


def test_qnetwork_forward():
    """Test QNetwork forward pass."""
    try:
        net = QNetwork(state_dim=4, action_dim=2, hidden_dim=16)

        # Single state
        state = torch.randn(4)
        q_values = net(state.unsqueeze(0))
        assert q_values.shape == (1, 2), f"Expected shape (1, 2), got {q_values.shape}"

        # Batch of states
        states = torch.randn(32, 4)
        q_values = net(states)
        assert q_values.shape == (
            32,
            2,
        ), f"Expected shape (32, 2), got {q_values.shape}"

        test_passed("qnetwork_forward")
        return True

    except Exception as e:
        test_failed("qnetwork_forward", str(e))
        return False


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("DQN Components Test Suite")
    print("=" * 60)
    print()

    all_tests = [
        (
            "Part 1: Replay Buffer",
            [
                test_replay_buffer,
                test_replay_buffer_capacity,
                test_nstep_buffer_basic,
                test_nstep_buffer_episode_boundary,
                test_nstep_buffer_one_step,
                test_batch_to_tensors,
            ],
        ),
        (
            "Part 2: Epsilon-Greedy Exploration",
            [
                test_epsilon_greedy_greedy,
                test_epsilon_greedy_random,
                test_epsilon_greedy_2d_input,
                test_linear_epsilon_decay,
            ],
        ),
        (
            "Part 3: TD Target Computation",
            [
                test_compute_td_target_basic,
                test_compute_td_target_terminal,
                test_compute_td_target_detach,
                test_compute_double_dqn_target,
                test_compute_double_dqn_target_terminal,
                test_compute_double_dqn_target_detach,
            ],
        ),
        (
            "Part 4: TD Loss Computation",
            [
                test_compute_td_loss_huber,
                test_compute_td_loss_nonzero,
                test_compute_td_loss_gradient,
                test_compute_td_loss_detach,
            ],
        ),
        (
            "Part 5: Target Network Updates",
            [
                test_soft_update,
                test_soft_update_tau_one,
                test_hard_update,
            ],
        ),
        (
            "Part 6: Q-Network",
            [
                test_qnetwork_forward,
            ],
        ),
    ]

    total_passed = 0
    total_tests = 0

    for part_name, tests in all_tests:
        print(f"{part_name}")
        print("-" * 40)
        for test_fn in tests:
            total_tests += 1
            if test_fn():
                total_passed += 1
        print()

    print("=" * 60)
    if total_passed == total_tests:
        print(f"{Colors.GREEN}All {total_tests} tests passed!{Colors.RESET}")
        print("""
      _          ___
    /' '\\       / " \\
   |  ,--+-----4 /   |
   ',/   o  o     --.;
--._|_   ,--.  _.,-- \\----.
------'--`--' '-----,' VJ  |
     \\_  ._\\_.   _,-'---._.'
       `--...--``  /
         /###\\   | |
         |.   `.-'-'.
        .||  /,     |
       do_o00oo_,.ob

Here's a puppy video as a reward: https://openpuppies.com/mp4/P6Q0XzB.mp4
""")
    else:
        print(f"{Colors.RED}{total_passed}/{total_tests} tests passed{Colors.RESET}")
    print("=" * 60)


if __name__ == "__main__":
    main()
