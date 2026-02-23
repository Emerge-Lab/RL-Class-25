#!/usr/bin/env python3
"""
PPO Component Tests - Homework 2, Problem 1

Testing framework inspired by Sasha Rush's Tensor Puzzles:
https://github.com/srush/Tensor-Puzzles

Run with: uv run python test_components.py
"""

import torch
from typing import Callable, List, Tuple, Dict, Any
import traceback
import sys
import random

# Import student implementations
from ppo_components import (
    compute_returns,
    compute_gae,
    compute_policy_loss,
    compute_value_loss,
    compute_entropy_bonus,
    normalize_advantages,
    discrete_log_prob,
    discrete_entropy,
    gaussian_log_prob,
    gaussian_entropy,
    sample_discrete_action,
    sample_continuous_action,
    squashed_gaussian_log_prob,
    clip_action,
    RolloutBuffer,
)


# =============================================================================
# Test Framework (inspired by Tensor Puzzles)
# =============================================================================


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


def run_test(
    name: str,
    fn: Callable,
    test_cases: List[Tuple[Dict[str, Any], torch.Tensor]],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> TestResult:
    """
    Run a test on a function with multiple test cases.
    """
    try:
        for i, (inputs, expected) in enumerate(test_cases):
            result = fn(**inputs)

            if not torch.is_tensor(result):
                return TestResult(
                    name, False, f"Case {i+1}: Expected tensor, got {type(result)}"
                )

            if result.shape != expected.shape:
                return TestResult(
                    name,
                    False,
                    f"Case {i+1}: Shape mismatch. Expected {expected.shape}, got {result.shape}",
                )

            if not torch.allclose(result, expected, rtol=rtol, atol=atol):
                diff = (result - expected).abs().max().item()
                return TestResult(
                    name,
                    False,
                    f"Case {i+1}: Values don't match. Max diff: {diff:.6f}\n"
                    f"  Expected: {expected}\n"
                    f"  Got:      {result}",
                )

        return TestResult(name, True, f"All {len(test_cases)} test cases passed!")

    except NotImplementedError:
        return TestResult(name, False, "Not implemented yet")
    except Exception as e:
        return TestResult(name, False, f"Error: {e}\n{traceback.format_exc()}")


# =============================================================================
# Part 1: Core RL Computation Tests
# =============================================================================


def test_compute_returns() -> TestResult:
    """Test the compute_returns function."""
    test_cases = [
        # Case 1: Simple trajectory, no episode boundaries (num_steps=4, num_envs=2)
        (
            {
                "rewards": torch.tensor(
                    [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
                ),
                "dones": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                "gamma": 0.99,
            },
            torch.tensor(
                [
                    [
                        1.0 + 0.99 * (1.0 + 0.99 * (1.0 + 0.99 * 1.0)),
                        2.0 + 0.99 * (2.0 + 0.99 * (2.0 + 0.99 * 2.0)),
                    ],
                    [1.0 + 0.99 * (1.0 + 0.99 * 1.0), 2.0 + 0.99 * (2.0 + 0.99 * 2.0)],
                    [1.0 + 0.99 * 1.0, 2.0 + 0.99 * 2.0],
                    [1.0, 2.0],
                ]
            ),
        ),
        # Case 2: Episode boundary in env 0 but not env 1 (num_steps=4, num_envs=2)
        (
            {
                "rewards": torch.tensor(
                    [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]
                ),
                "dones": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                "gamma": 0.9,
            },
            torch.tensor(
                [
                    [1.0 + 0.9 * 2.0, 1.0 + 0.9 * (1.0 + 0.9 * (1.0 + 0.9 * 1.0))],
                    [2.0, 1.0 + 0.9 * (1.0 + 0.9 * 1.0)],
                    [3.0 + 0.9 * 4.0, 1.0 + 0.9 * 1.0],
                    [4.0, 1.0],
                ]
            ),
        ),
        # Case 3: gamma = 0 (no discounting of future rewards)
        (
            {
                "rewards": torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
                "dones": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                "gamma": 0.0,
            },
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ),
    ]
    return run_test("compute_returns", compute_returns, test_cases)


def test_compute_gae() -> TestResult:
    """Test the compute_gae function."""
    test_cases = [
        # Case 1: Simple GAE calculation (num_steps=3, num_envs=2)
        # Env 0: rewards=1, values=0.5 -> delta = 1 + 0.99*0.5 - 0.5 = 0.995
        # Env 1: rewards=2, values=1.0 -> delta = 2 + 0.99*1.0 - 1.0 = 1.99
        (
            {
                "rewards": torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
                "values": torch.tensor(
                    [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0], [0.5, 1.0]]
                ),
                "dones": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                "gamma": 0.99,
                "lam": 0.95,
            },
            torch.tensor(
                [
                    [
                        0.995 + 0.9405 * (0.995 + 0.9405 * 0.995),
                        1.99 + 0.9405 * (1.99 + 0.9405 * 1.99),
                    ],
                    [0.995 + 0.9405 * 0.995, 1.99 + 0.9405 * 1.99],
                    [0.995, 1.99],
                ]
            ),
        ),
        # Case 2: Episode boundary in env 0 but not env 1 (num_steps=3, num_envs=2)
        (
            {
                "rewards": torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]),
                "values": torch.tensor(
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
                ),
                "dones": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
                "gamma": 0.9,
                "lam": 0.8,
            },
            torch.tensor(
                [
                    [1.62, 0.9 + 0.72 * (0.9 + 0.72 * 0.9)],
                    [1.0, 0.9 + 0.72 * 0.9],
                    [0.9, 0.9],
                ]
            ),
        ),
    ]
    return run_test("compute_gae", compute_gae, test_cases, rtol=1e-3)


def test_normalize_advantages() -> TestResult:
    """Test the normalize_advantages function."""
    test_cases = [
        # Case 1: Simple normalization
        (
            {
                "advantages": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            },
            torch.tensor([-1.2649, -0.6325, 0.0, 0.6325, 1.2649]),
        ),
        # Case 2: Already centered (mean=0)
        (
            {
                "advantages": torch.tensor([-1.0, 0.0, 1.0]),
            },
            torch.tensor([-1.0, 0.0, 1.0]),
        ),
        # Case 3: All identical (zero std — epsilon prevents division by zero)
        (
            {
                "advantages": torch.tensor([5.0, 5.0, 5.0]),
            },
            torch.tensor([0.0, 0.0, 0.0]),
        ),
    ]
    return run_test("normalize_advantages", normalize_advantages, test_cases, rtol=1e-3)


# =============================================================================
# Part 2: Policy Distribution Tests
# =============================================================================


def test_discrete_log_prob() -> TestResult:
    """Test the discrete_log_prob function."""
    test_cases = [
        # Case 1: Uniform logits
        (
            {
                "logits": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                "actions": torch.tensor([0]),
            },
            torch.tensor([-1.3863]),  # log(0.25)
        ),
        # Case 2: Peaked distribution
        (
            {
                "logits": torch.tensor([[10.0, 0.0, 0.0]]),
                "actions": torch.tensor([0]),
            },
            # After softmax, p[0] ≈ 1, so log(p[0]) ≈ 0
            torch.tensor([0.0]),
        ),
        # Case 3: Batch of actions
        (
            {
                "logits": torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
                "actions": torch.tensor([1, 0]),
            },
            # log_softmax([1,2])[1] = 2 - log(e^1 + e^2) = 2 - log(e^1(1+e)) ≈ -0.3133
            # log_softmax([3,1])[0] = 3 - log(e^3 + e^1) = 3 - log(e^1(e^2+1)) ≈ -0.1269
            torch.tensor([-0.3133, -0.1269]),
        ),
    ]
    return run_test(
        "discrete_log_prob", discrete_log_prob, test_cases, rtol=1e-3, atol=1e-3
    )


def test_discrete_entropy() -> TestResult:
    """Test the discrete_entropy function."""
    test_cases = [
        # Case 1: Uniform distribution (maximum entropy)
        (
            {
                "logits": torch.tensor([[0.0, 0.0]]),
            },
            torch.tensor([0.6931]),  # log(2)
        ),
        # Case 2: Peaked distribution (low entropy)
        (
            {
                "logits": torch.tensor([[100.0, 0.0]]),
            },
            torch.tensor([0.0]),  # Nearly deterministic
        ),
        # Case 3: Batch
        (
            {
                "logits": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            },
            torch.tensor([1.0986, 1.0986]),  # log(3) for each
        ),
    ]
    return run_test(
        "discrete_entropy", discrete_entropy, test_cases, rtol=1e-3, atol=1e-3
    )


def test_gaussian_log_prob() -> TestResult:
    """Test the gaussian_log_prob function."""
    test_cases = [
        # Case 1: Standard normal, action at mean
        (
            {
                "mean": torch.tensor([[0.0]]),
                "log_std": torch.tensor([0.0]),  # std = 1
                "actions": torch.tensor([[0.0]]),
            },
            # log prob = -0.5 * (0^2 + 0 + log(2*pi)) = -0.5 * log(2*pi) ≈ -0.9189
            torch.tensor([-0.9189]),
        ),
        # Case 2: Action 1 std away from mean
        (
            {
                "mean": torch.tensor([[0.0]]),
                "log_std": torch.tensor([0.0]),
                "actions": torch.tensor([[1.0]]),
            },
            # log prob = -0.5 * (1 + 0 + log(2*pi)) ≈ -1.4189
            torch.tensor([-1.4189]),
        ),
        # Case 3: Multi-dimensional action
        (
            {
                "mean": torch.tensor([[0.0, 0.0]]),
                "log_std": torch.tensor([0.0, 0.0]),
                "actions": torch.tensor([[0.0, 0.0]]),
            },
            # Sum of two independent Gaussians at mean
            torch.tensor([-1.8379]),
        ),
    ]
    return run_test(
        "gaussian_log_prob", gaussian_log_prob, test_cases, rtol=1e-3, atol=1e-3
    )


def test_gaussian_entropy() -> TestResult:
    """Test the gaussian_entropy function."""
    test_cases = [
        # Case 1: Standard normal (1D)
        (
            {
                "log_std": torch.tensor([0.0]),
            },
            # H = 0.5 + 0.5*log(2*pi) + 0 = 0.5 + 0.9189 = 1.4189
            torch.tensor(1.4189),
        ),
        # Case 2: Higher variance
        (
            {
                "log_std": torch.tensor([1.0]),  # std = e
            },
            # H = 0.5 + 0.5*log(2*pi) + 1 = 2.4189
            torch.tensor(2.4189),
        ),
        # Case 3: Multi-dimensional
        (
            {
                "log_std": torch.tensor([0.0, 0.0]),
            },
            # Sum of two independent Gaussians
            torch.tensor(2.8379),
        ),
    ]
    return run_test(
        "gaussian_entropy", gaussian_entropy, test_cases, rtol=1e-3, atol=1e-3
    )


# =============================================================================
# Part 2b: Action Sampling Tests
# =============================================================================


def test_sample_discrete_action() -> TestResult:
    """Test discrete action sampling."""
    try:
        torch.manual_seed(42)

        # Test 1: Shape check
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        actions, log_probs = sample_discrete_action(logits)

        if actions.shape != (2,):
            return TestResult(
                "sample_discrete_action",
                False,
                f"Actions shape wrong: expected (2,), got {actions.shape}",
            )
        if log_probs.shape != (2,):
            return TestResult(
                "sample_discrete_action",
                False,
                f"Log probs shape wrong: expected (2,), got {log_probs.shape}",
            )

        # Test 2: Actions should be valid indices
        if not (actions >= 0).all() or not (actions < 3).all():
            return TestResult(
                "sample_discrete_action",
                False,
                f"Actions out of range [0, 2]: {actions}",
            )

        # Test 3: Log probs should be negative (probabilities < 1)
        if not (log_probs <= 0).all():
            return TestResult(
                "sample_discrete_action",
                False,
                f"Log probs should be <= 0: {log_probs}",
            )

        # Test 4: Peaked distribution should usually sample the high-prob action
        torch.manual_seed(0)
        peaked_logits = torch.tensor([[100.0, 0.0, 0.0]])
        successes = 0
        for _ in range(10):
            action, _ = sample_discrete_action(peaked_logits)
            if action.item() == 0:
                successes += 1
        if successes < 9:  # Should be action 0 almost always
            return TestResult(
                "sample_discrete_action",
                False,
                "Peaked distribution should sample action 0 most of the time",
            )

        return TestResult("sample_discrete_action", True, "All tests passed!")

    except NotImplementedError:
        return TestResult("sample_discrete_action", False, "Not implemented yet")
    except Exception as e:
        return TestResult(
            "sample_discrete_action", False, f"Error: {e}\n{traceback.format_exc()}"
        )


def test_sample_continuous_action() -> TestResult:
    """Test continuous action sampling with tanh squashing."""
    try:
        torch.manual_seed(42)

        # Test 1: Shape check
        mean = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        log_std = torch.tensor([0.0, 0.0])
        actions, log_probs = sample_continuous_action(mean, log_std)

        if actions.shape != (2, 2):
            return TestResult(
                "sample_continuous_action",
                False,
                f"Actions shape wrong: expected (2, 2), got {actions.shape}",
            )
        if log_probs.shape != (2,):
            return TestResult(
                "sample_continuous_action",
                False,
                f"Log probs shape wrong: expected (2,), got {log_probs.shape}",
            )

        # Test 2: Actions should be bounded to [-1, 1] due to tanh
        if not (actions >= -1.0).all() or not (actions <= 1.0).all():
            return TestResult(
                "sample_continuous_action",
                False,
                f"Actions should be in [-1, 1] due to tanh: min={actions.min()}, max={actions.max()}",
            )

        # Test 3: High variance should still produce bounded actions
        high_var_log_std = torch.tensor([2.0, 2.0])  # std = e^2 ≈ 7.4
        actions_high_var, _ = sample_continuous_action(mean, high_var_log_std)
        if not (actions_high_var >= -1.0).all() or not (actions_high_var <= 1.0).all():
            return TestResult(
                "sample_continuous_action",
                False,
                "High variance actions should still be bounded to [-1, 1]",
            )

        # Test 4: Log probs should be finite
        if not torch.isfinite(log_probs).all():
            return TestResult(
                "sample_continuous_action",
                False,
                f"Log probs should be finite: {log_probs}",
            )

        return TestResult("sample_continuous_action", True, "All tests passed!")

    except NotImplementedError:
        return TestResult("sample_continuous_action", False, "Not implemented yet")
    except Exception as e:
        return TestResult(
            "sample_continuous_action", False, f"Error: {e}\n{traceback.format_exc()}"
        )


def test_squashed_gaussian_log_prob() -> TestResult:
    """Test log probability computation for squashed (tanh) actions."""
    try:
        # Test 1: Action at the mean (before squashing)
        # If mean=0, std=1, and we sample z=0, then action=tanh(0)=0
        # The log prob should equal gaussian_log_prob at z=0 minus Jacobian correction
        mean = torch.tensor([[0.0]])
        log_std = torch.tensor([0.0])
        action = torch.tensor([[0.0]])  # tanh(0) = 0

        log_prob = squashed_gaussian_log_prob(mean, log_std, action)

        if log_prob.shape != (1,):
            return TestResult(
                "squashed_gaussian_log_prob",
                False,
                f"Log prob shape wrong: expected (1,), got {log_prob.shape}",
            )

        # At action=0, Jacobian correction = log(1 - 0^2) = 0
        # So log_prob should equal gaussian_log_prob at z=0
        expected = gaussian_log_prob(mean, log_std, action)
        if not torch.allclose(log_prob, expected, atol=1e-3):
            return TestResult(
                "squashed_gaussian_log_prob",
                False,
                f"At action=0, log prob should equal unbounded: {log_prob} vs {expected}",
            )

        # Test 2: Action near boundary (large Jacobian correction)
        action_near_boundary = torch.tensor([[0.9]])
        log_prob_boundary = squashed_gaussian_log_prob(
            mean, log_std, action_near_boundary
        )

        # Jacobian correction at 0.9: log(1 - 0.81) = log(0.19) ≈ -1.66
        # This should make log_prob more negative (higher density from squashing)
        z = torch.atanh(action_near_boundary.clamp(-0.999, 0.999))
        log_prob_z = gaussian_log_prob(mean, log_std, z)
        jacobian_correction = torch.log(1 - action_near_boundary.pow(2) + 1e-6).sum(
            dim=-1
        )
        expected_boundary = log_prob_z - jacobian_correction

        if not torch.allclose(log_prob_boundary, expected_boundary, atol=1e-3):
            return TestResult(
                "squashed_gaussian_log_prob",
                False,
                f"Jacobian correction incorrect: {log_prob_boundary} vs {expected_boundary}",
            )

        # Test 3: Batch processing
        batch_mean = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        batch_log_std = torch.tensor([0.0, 0.0])
        batch_action = torch.tensor([[0.0, 0.5], [0.7, -0.3]])
        batch_log_prob = squashed_gaussian_log_prob(
            batch_mean, batch_log_std, batch_action
        )

        if batch_log_prob.shape != (2,):
            return TestResult(
                "squashed_gaussian_log_prob",
                False,
                f"Batch log prob shape wrong: expected (2,), got {batch_log_prob.shape}",
            )

        return TestResult("squashed_gaussian_log_prob", True, "All tests passed!")

    except NotImplementedError:
        return TestResult("squashed_gaussian_log_prob", False, "Not implemented yet")
    except Exception as e:
        return TestResult(
            "squashed_gaussian_log_prob", False, f"Error: {e}\n{traceback.format_exc()}"
        )


def test_clip_action() -> TestResult:
    """Test action clipping function."""
    test_cases = [
        # Case 1: Values already in range (no clipping needed)
        (
            {
                "action": torch.tensor([[0.0, 0.5, -0.5]]),
                "low": -1.0,
                "high": 1.0,
            },
            torch.tensor([[0.0, 0.5, -0.5]]),
        ),
        # Case 2: Values outside range (clipping needed)
        (
            {
                "action": torch.tensor([[-2.0, 0.0, 2.0]]),
                "low": -1.0,
                "high": 1.0,
            },
            torch.tensor([[-1.0, 0.0, 1.0]]),
        ),
        # Case 3: Custom range
        (
            {
                "action": torch.tensor([[0.0, 5.0, 15.0]]),
                "low": 0.0,
                "high": 10.0,
            },
            torch.tensor([[0.0, 5.0, 10.0]]),
        ),
        # Case 4: Batch of actions
        (
            {
                "action": torch.tensor([[-5.0, 0.0], [0.0, 5.0]]),
                "low": -1.0,
                "high": 1.0,
            },
            torch.tensor([[-1.0, 0.0], [0.0, 1.0]]),
        ),
    ]
    return run_test("clip_action", clip_action, test_cases)


# =============================================================================
# Part 3: PPO Loss Function Tests
# =============================================================================


def test_compute_policy_loss() -> TestResult:
    """Test the compute_policy_loss function."""
    test_cases = [
        # Case 1: No clipping needed (ratio within bounds)
        (
            {
                "log_probs": torch.tensor([-1.0, -1.0, -1.0]),
                "old_log_probs": torch.tensor([-1.0, -1.0, -1.0]),
                "advantages": torch.tensor([1.0, 2.0, 3.0]),
                "clip_epsilon": 0.2,
            },
            torch.tensor(-2.0),
        ),
        # Case 2: Clipping activated (ratio too high with positive advantage)
        (
            {
                "log_probs": torch.tensor([0.0]),
                "old_log_probs": torch.tensor([-1.0]),
                "advantages": torch.tensor([1.0]),
                "clip_epsilon": 0.2,
            },
            torch.tensor(-1.2),
        ),
        # Case 3: Clipping with negative advantage
        (
            {
                "log_probs": torch.tensor([-2.0]),
                "old_log_probs": torch.tensor([0.0]),
                "advantages": torch.tensor([-1.0]),
                "clip_epsilon": 0.2,
            },
            torch.tensor(0.8),
        ),
    ]
    return run_test("compute_policy_loss", compute_policy_loss, test_cases, rtol=1e-3)


def test_compute_value_loss() -> TestResult:
    """Test the compute_value_loss function."""
    test_cases = [
        # Case 1: Perfect predictions
        (
            {
                "values": torch.tensor([1.0, 2.0, 3.0]),
                "returns": torch.tensor([1.0, 2.0, 3.0]),
            },
            torch.tensor(0.0),
        ),
        # Case 2: MSE calculation
        (
            {
                "values": torch.tensor([0.0, 0.0, 0.0]),
                "returns": torch.tensor([1.0, 2.0, 3.0]),
            },
            torch.tensor(14.0 / 3.0),
        ),
        # Case 3: Negative values
        (
            {
                "values": torch.tensor([-1.0, 0.0, 1.0]),
                "returns": torch.tensor([1.0, 0.0, -1.0]),
            },
            torch.tensor(8.0 / 3.0),
        ),
    ]
    return run_test("compute_value_loss", compute_value_loss, test_cases)


def test_compute_entropy_bonus() -> TestResult:
    """Test the compute_entropy_bonus function."""
    test_cases = [
        # Case 1: Uniform distribution (maximum entropy)
        (
            {
                "probs": torch.tensor([[0.5, 0.5]]),
            },
            torch.tensor(0.6931471805599453),
        ),
        # Case 2: Deterministic distribution (zero entropy)
        (
            {
                "probs": torch.tensor([[1.0, 0.0]]),
            },
            torch.tensor(0.0),
        ),
        # Case 3: Batch of distributions
        (
            {
                "probs": torch.tensor(
                    [
                        [0.5, 0.5],
                        [0.25, 0.75],
                    ]
                ),
            },
            torch.tensor(0.6278982),
        ),
    ]
    return run_test(
        "compute_entropy_bonus", compute_entropy_bonus, test_cases, rtol=1e-3, atol=1e-3
    )


# =============================================================================
# Part 4: Rollout Buffer Test
# =============================================================================


def test_rollout_buffer() -> TestResult:
    """Test the RolloutBuffer class."""
    try:
        # Create buffer
        buffer = RolloutBuffer(
            num_steps=4,
            num_envs=2,
            obs_shape=(3,),
            action_shape=(),
            device=torch.device("cpu"),
        )

        # Add some data
        for step in range(4):
            buffer.add(
                obs=torch.randn(2, 3),
                action=torch.randint(0, 4, (2,)),
                log_prob=torch.randn(2),
                reward=torch.ones(2) * step,
                done=torch.zeros(2),
                value=torch.ones(2) * 0.5,
            )

        # Compute returns and advantages
        last_value = torch.ones(2) * 0.5
        last_done = torch.zeros(2)
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Check shapes
        if returns.shape != (4, 2):
            return TestResult(
                "RolloutBuffer", False, f"Returns shape wrong: {returns.shape}"
            )
        if advantages.shape != (4, 2):
            return TestResult(
                "RolloutBuffer", False, f"Advantages shape wrong: {advantages.shape}"
            )

        # Check that returns = advantages + values
        expected_returns = advantages + buffer.values
        if not torch.allclose(returns, expected_returns):
            return TestResult("RolloutBuffer", False, "returns != advantages + values")

        # Test batching: verify correct count and shapes
        batch_count = 0
        for batch in buffer.get_batches(
            batch_size=4, returns=returns, advantages=advantages
        ):
            batch_count += 1
            if batch["obs"].shape[0] != 4:
                return TestResult(
                    "RolloutBuffer",
                    False,
                    f"Batch obs shape wrong: {batch['obs'].shape}",
                )

        if batch_count != 2:  # 8 total samples / 4 batch size = 2 batches
            return TestResult(
                "RolloutBuffer", False, f"Expected 2 batches, got {batch_count}"
            )

        # Test batching: verify all samples appear exactly once
        torch.manual_seed(0)
        all_values = []
        for batch in buffer.get_batches(
            batch_size=4, returns=returns, advantages=advantages
        ):
            all_values.append(batch["values"])
        all_values = torch.cat(all_values)
        flat_values = buffer.values.reshape(-1)
        # Sort both and compare — same elements, just shuffled
        if not torch.allclose(all_values.sort()[0], flat_values.sort()[0]):
            return TestResult(
                "RolloutBuffer",
                False,
                "get_batches doesn't return all samples exactly once",
            )

        return TestResult("RolloutBuffer", True, "All buffer tests passed!")

    except NotImplementedError:
        return TestResult("RolloutBuffer", False, "Not implemented yet")
    except Exception as e:
        return TestResult(
            "RolloutBuffer", False, f"Error: {e}\n{traceback.format_exc()}"
        )


def test_rollout_buffer_episode_boundary() -> TestResult:
    """Test RolloutBuffer with episode boundaries (dones=1 mid-rollout and last_done=1)."""
    try:
        buffer = RolloutBuffer(
            num_steps=4,
            num_envs=1,
            obs_shape=(2,),
            action_shape=(),
            device=torch.device("cpu"),
        )

        # Rewards: [1, 1, 1, 1], dones: [0, 1, 0, 0]
        # Episode ends at step 1, new episode starts at step 2
        rewards = [1.0, 1.0, 1.0, 1.0]
        dones = [0.0, 1.0, 0.0, 0.0]
        value = 0.5

        for step in range(4):
            buffer.add(
                obs=torch.randn(1, 2),
                action=torch.randint(0, 2, (1,)),
                log_prob=torch.randn(1),
                reward=torch.tensor([rewards[step]]),
                done=torch.tensor([dones[step]]),
                value=torch.tensor([value]),
            )

        # Test with last_done=1 (episode just ended at the final step)
        last_value = torch.tensor([10.0])  # Large value that should be zeroed out
        last_done = torch.tensor([1.0])

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Check shapes
        if returns.shape != (4, 1):
            return TestResult(
                "RolloutBuffer_episode_boundary",
                False,
                f"Returns shape wrong: {returns.shape}",
            )

        # Key invariant: returns = advantages + values
        expected_returns = advantages + buffer.values
        if not torch.allclose(returns, expected_returns):
            return TestResult(
                "RolloutBuffer_episode_boundary",
                False,
                "returns != advantages + values",
            )

        # With last_done=1, bootstrap value should be 0, not 10.0
        # So the advantage at step 3 should be:
        # delta_3 = reward_3 + gamma * 0 - value_3 = 1.0 + 0 - 0.5 = 0.5
        # gae_3 = delta_3 = 0.5
        expected_adv_3 = torch.tensor([0.5])
        if not torch.allclose(advantages[3, :], expected_adv_3, atol=1e-4):
            return TestResult(
                "RolloutBuffer_episode_boundary",
                False,
                f"Advantage at step 3 with last_done=1 should be 0.5, got {advantages[3, :].item():.4f}. "
                f"Is last_done being used to zero out the bootstrap value?",
            )

        # With done=1 at step 1, the advantage at step 1 should only use
        # its own reward (no bootstrap from future):
        # delta_1 = reward_1 + gamma * value_2 * (1 - done_1) - value_1
        #         = 1.0 + 0.99 * 0.5 * 0 - 0.5 = 0.5
        expected_adv_1 = torch.tensor([0.5])
        if not torch.allclose(advantages[1, :], expected_adv_1, atol=1e-4):
            return TestResult(
                "RolloutBuffer_episode_boundary",
                False,
                f"Advantage at step 1 (terminal) should be 0.5, got {advantages[1, :].item():.4f}",
            )

        return TestResult(
            "RolloutBuffer_episode_boundary", True, "Episode boundary tests passed!"
        )

    except NotImplementedError:
        return TestResult(
            "RolloutBuffer_episode_boundary", False, "Not implemented yet"
        )
    except Exception as e:
        return TestResult(
            "RolloutBuffer_episode_boundary",
            False,
            f"Error: {e}\n{traceback.format_exc()}",
        )


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests() -> None:
    """Run all tests and display results."""
    print("=" * 60)
    print("PPO Components Test Suite")
    print("=" * 60)
    print()

    # Group tests by part
    parts = [
        (
            "Part 1: Core RL Computations",
            [
                test_compute_returns,
                test_compute_gae,
                test_normalize_advantages,
            ],
        ),
        (
            "Part 2a: Policy Distributions",
            [
                test_discrete_log_prob,
                test_discrete_entropy,
                test_gaussian_log_prob,
                test_gaussian_entropy,
            ],
        ),
        (
            "Part 2b: Action Sampling",
            [
                test_sample_discrete_action,
                test_sample_continuous_action,
                test_squashed_gaussian_log_prob,
                test_clip_action,
            ],
        ),
        (
            "Part 3: PPO Loss Functions",
            [
                test_compute_policy_loss,
                test_compute_value_loss,
                test_compute_entropy_bonus,
            ],
        ),
        (
            "Part 4: Rollout Buffer",
            [
                test_rollout_buffer,
                test_rollout_buffer_episode_boundary,
            ],
        ),
    ]

    results = []
    for part_name, tests in parts:
        print(f"\n{part_name}")
        print("-" * 40)
        for test_fn in tests:
            result = test_fn()
            results.append(result)

            status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
            print(f"  [{status}] {result.name}")
            if not result.passed:
                for line in result.message.split("\n"):
                    print(f"         {line}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    if passed == total:
        print(f"\033[92mAll {total} tests passed!\033[0m")
        print()
        # Celebrate with a puppy! (inspired by Tensor Puzzles)
        pups = [
            "2m78jPG",
            "pn1e9TO",
            "MQCIwzT",
            "udLK6FS",
            "ZNem5o3",
            "DS2IZ6K",
            "aydRUz8",
            "MVUdQYK",
            "kLvno0p",
            "wScLiVz",
            "Z0TII8i",
            "F1SChho",
            "9hRi2jN",
            "lvzRF3W",
            "fqHxOGI",
            "1xeUYme",
            "6tVqKyM",
            "CCxZ6Wr",
        ]
        pup = random.choice(pups)
        print("      _          ___          ")
        print("    /' '\\       / \" \\         ")
        print("   |  ,--+-----4 /   |        ")
        print("   ',/   o  o     --.;        ")
        print("--._|_   ,--.  _.,-- \\----.   ")
        print("------'--`--' '-----,' VJ  |  ")
        print("     \\_  ._\\_.   _,-'---._.'  ")
        print("       `--...--``  /          ")
        print("         /###\\   | |          ")
        print("         |.   `.-'-'.         ")
        print("        .||  /,     |         ")
        print("       do_o00oo_,.ob          ")
        print()
        print(
            f"Here's a puppy video as a reward: https://openpuppies.com/mp4/{pup}.mp4"
        )
    else:
        print(f"\033[93m{passed}/{total} tests passed\033[0m")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_all_tests()
