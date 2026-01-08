"""Tests for the evaluation module."""

import tempfile
from pathlib import Path

import pytest
import torch

from server.evaluate import (
    EvalResult,
    load_policy_from_submission,
    evaluate_policy,
    evaluate_submission,
)


@pytest.fixture
def simple_policy_dir():
    """Create a temporary directory with a simple policy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.py"
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        # Write a simple policy that always returns action 0
        policy_path.write_text('''
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Always push left (action 0)
        if x.dim() == 1:
            return torch.tensor(0)
        return torch.zeros(x.shape[0], dtype=torch.long)

def load_policy(checkpoint_path):
    return Policy()
''')

        # Write empty checkpoint
        torch.save({}, checkpoint_path)

        yield Path(tmpdir)


@pytest.fixture
def trained_policy_dir():
    """Create a temporary directory with a trained policy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.py"
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        # Write a policy that tries to balance
        policy_path.write_text('''
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x):
        # Simple heuristic: push in direction of pole tilt
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # x[2] is pole angle - push right if tilting right
        action = (x[:, 2] > 0).long()
        if action.shape[0] == 1:
            return action.item()
        return action

def load_policy(checkpoint_path):
    return Policy()
''')

        torch.save({}, checkpoint_path)
        yield Path(tmpdir)


class TestEvalResult:
    def test_eval_result_creation(self):
        result = EvalResult(
            mean_reward=100.0,
            std_reward=10.0,
            mean_length=50.0,
            episodes=100,
            eval_time=5.0,
        )
        assert result.mean_reward == 100.0
        assert result.std_reward == 10.0
        assert result.mean_length == 50.0
        assert result.episodes == 100
        assert result.eval_time == 5.0


class TestLoadPolicy:
    def test_load_valid_policy(self, simple_policy_dir):
        policy_path = simple_policy_dir / "policy.py"
        checkpoint_path = simple_policy_dir / "checkpoint.pt"

        policy = load_policy_from_submission(policy_path, checkpoint_path)
        assert callable(policy)

        # Test that policy returns an action
        obs = torch.zeros(4)
        action = policy(obs)
        assert action == 0

    def test_missing_load_policy_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.py"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Write policy without load_policy function
            policy_path.write_text("x = 1")
            torch.save({}, checkpoint_path)

            with pytest.raises(ValueError, match="must define load_policy"):
                load_policy_from_submission(policy_path, checkpoint_path)


class TestEvaluatePolicy:
    def test_evaluate_simple_policy(self, simple_policy_dir):
        policy_path = simple_policy_dir / "policy.py"
        checkpoint_path = simple_policy_dir / "checkpoint.pt"
        policy = load_policy_from_submission(policy_path, checkpoint_path)

        result = evaluate_policy(
            policy,
            env_name="cartpole",
            num_episodes=5,
            seed=42,
            timeout=30.0,
        )

        assert isinstance(result, EvalResult)
        assert result.episodes == 5
        assert result.mean_reward > 0
        assert result.std_reward >= 0
        assert result.mean_length > 0
        assert result.eval_time > 0

    def test_evaluate_with_timeout(self, simple_policy_dir):
        policy_path = simple_policy_dir / "policy.py"
        checkpoint_path = simple_policy_dir / "checkpoint.pt"
        policy = load_policy_from_submission(policy_path, checkpoint_path)

        # Very short timeout - CartPole runs fast, so use extremely short timeout
        result = evaluate_policy(
            policy,
            env_name="cartpole",
            num_episodes=100000,  # Request many episodes
            seed=42,
            timeout=0.01,  # 10ms timeout
        )

        # Should complete fewer episodes due to timeout
        assert result.episodes < 100000

    def test_unknown_environment(self, simple_policy_dir):
        policy_path = simple_policy_dir / "policy.py"
        checkpoint_path = simple_policy_dir / "checkpoint.pt"
        policy = load_policy_from_submission(policy_path, checkpoint_path)

        with pytest.raises(ValueError, match="Unknown environment"):
            evaluate_policy(policy, env_name="unknown_env")

    def test_deterministic_with_seed(self, trained_policy_dir):
        policy_path = trained_policy_dir / "policy.py"
        checkpoint_path = trained_policy_dir / "checkpoint.pt"
        policy = load_policy_from_submission(policy_path, checkpoint_path)

        result1 = evaluate_policy(
            policy, env_name="cartpole", num_episodes=10, seed=42
        )

        # Reload policy to reset any state
        policy = load_policy_from_submission(policy_path, checkpoint_path)
        result2 = evaluate_policy(
            policy, env_name="cartpole", num_episodes=10, seed=42
        )

        assert result1.mean_reward == result2.mean_reward
        assert result1.mean_length == result2.mean_length


class TestEvaluateSubmission:
    def test_evaluate_submission_direct(self, simple_policy_dir):
        result = evaluate_submission(
            simple_policy_dir,
            env_name="cartpole",
            num_episodes=5,
            seed=42,
            timeout=30.0,
            sandboxed=False,
        )

        assert isinstance(result, EvalResult)
        assert result.episodes == 5

    def test_missing_policy_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({}, checkpoint_path)

            with pytest.raises(FileNotFoundError, match="Missing policy.py"):
                evaluate_submission(Path(tmpdir), sandboxed=False)

    def test_missing_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.py"
            policy_path.write_text("def load_policy(p): pass")

            with pytest.raises(FileNotFoundError, match="Missing checkpoint.pt"):
                evaluate_submission(Path(tmpdir), sandboxed=False)
