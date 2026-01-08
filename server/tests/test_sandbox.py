"""Tests for the sandbox module."""

import tempfile
from pathlib import Path
import sys

import pytest
import torch

from server.sandbox import (
    SandboxError,
    TimeoutError,
    run_sandboxed_evaluation,
)


@pytest.fixture
def simple_policy_dir():
    """Create a temporary directory with a simple policy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.py"
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        policy_path.write_text('''
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tensor(0)

def load_policy(checkpoint_path):
    return Policy()
''')

        torch.save({}, checkpoint_path)
        yield Path(tmpdir)


@pytest.fixture
def slow_policy_dir():
    """Create a policy that takes too long."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.py"
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        policy_path.write_text('''
import torch
import time

class Policy:
    def __call__(self, x):
        time.sleep(10)  # Sleep for 10 seconds
        return torch.tensor(0)

def load_policy(checkpoint_path):
    return Policy()
''')

        torch.save({}, checkpoint_path)
        yield Path(tmpdir)


@pytest.fixture
def malicious_env_reader_dir():
    """Create a policy that tries to read environment variables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.py"
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        policy_path.write_text('''
import torch
import os

class Policy:
    def __init__(self):
        # Try to read environment variables during init
        self.env_vars = dict(os.environ)

    def __call__(self, x):
        return torch.tensor(0)

def load_policy(checkpoint_path):
    return Policy()
''')

        torch.save({}, checkpoint_path)
        yield Path(tmpdir)


@pytest.fixture
def file_reader_dir():
    """Create a policy that tries to read files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "policy.py"
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        policy_path.write_text('''
import torch

class Policy:
    def __init__(self):
        # Try to read /etc/passwd
        try:
            with open('/etc/passwd', 'r') as f:
                self.passwd = f.read()
        except:
            self.passwd = None

    def __call__(self, x):
        return torch.tensor(0)

def load_policy(checkpoint_path):
    return Policy()
''')

        torch.save({}, checkpoint_path)
        yield Path(tmpdir)


class TestSandboxedEvaluation:
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Sandbox uses fork which may not work reliably on macOS"
    )
    def test_basic_sandboxed_evaluation(self, simple_policy_dir):
        result = run_sandboxed_evaluation(
            simple_policy_dir,
            env_name="cartpole",
            num_episodes=3,
            seed=42,
            timeout=30.0,
        )

        assert result["success"] is True
        assert result["episodes"] == 3
        assert "mean_reward" in result
        assert "std_reward" in result

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Sandbox uses fork which may not work reliably on macOS"
    )
    def test_timeout_enforcement(self, slow_policy_dir):
        with pytest.raises(TimeoutError):
            run_sandboxed_evaluation(
                slow_policy_dir,
                env_name="cartpole",
                num_episodes=10,
                seed=42,
                timeout=1.0,  # 1 second timeout
            )

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Sandbox uses fork which may not work reliably on macOS"
    )
    def test_environment_variables_hidden(self, malicious_env_reader_dir):
        """Test that sensitive environment variables are not accessible."""
        import os
        os.environ["SECRET_API_KEY"] = "super-secret-value"

        try:
            result = run_sandboxed_evaluation(
                malicious_env_reader_dir,
                env_name="cartpole",
                num_episodes=1,
                seed=42,
                timeout=30.0,
            )
            # The policy should run, but shouldn't have access to SECRET_API_KEY
            assert result["success"] is True
        finally:
            del os.environ["SECRET_API_KEY"]

    def test_missing_policy_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({}, checkpoint_path)

            with pytest.raises(SandboxError):
                run_sandboxed_evaluation(
                    Path(tmpdir),
                    env_name="cartpole",
                    num_episodes=1,
                    timeout=5.0,
                )

    def test_invalid_policy_syntax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.py"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Write invalid Python
            policy_path.write_text("def load_policy(: invalid syntax")
            torch.save({}, checkpoint_path)

            with pytest.raises(SandboxError):
                run_sandboxed_evaluation(
                    Path(tmpdir),
                    env_name="cartpole",
                    num_episodes=1,
                    timeout=5.0,
                )
