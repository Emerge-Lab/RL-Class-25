"""Evaluation runner for student policy submissions."""

import importlib.util
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EvalResult:
    mean_reward: float
    std_reward: float
    mean_length: float
    episodes: int
    eval_time: float


def load_policy_from_submission(policy_path: Path, checkpoint_path: Path):
    """
    Load a student's policy from their submission.

    The policy.py must define: load_policy(checkpoint_path) -> callable
    The returned callable must accept observations and return actions.
    """
    # Load the policy module dynamically
    spec = importlib.util.spec_from_file_location("student_policy", policy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_policy"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "load_policy"):
        raise ValueError("policy.py must define load_policy(checkpoint_path) function")

    policy = module.load_policy(checkpoint_path)
    return policy


def evaluate_policy(
    policy,
    env_name: str = "cartpole",
    num_episodes: int = 100,
    seed: int = 42,
    timeout: float = 60.0,
) -> EvalResult:
    """
    Evaluate a policy on a PufferLib environment.

    Args:
        policy: Callable that takes observations and returns actions
        env_name: Name of the PufferLib ocean environment
        num_episodes: Number of episodes to run
        seed: Random seed for reproducibility
        timeout: Maximum evaluation time in seconds

    Returns:
        EvalResult with statistics
    """
    # Import environment dynamically based on env_name
    if env_name == "cartpole":
        from pufferlib.ocean.cartpole.cartpole import Cartpole
        env = Cartpole(num_envs=1, render_mode=None)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    episode_rewards = []
    episode_lengths = []

    start_time = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_length = 0

    while len(episode_rewards) < num_episodes:
        # Check timeout
        if time.perf_counter() - start_time > timeout:
            break

        # Get action from policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            action = policy(obs_tensor)

            # Handle different action formats
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if hasattr(action, "__len__") and len(action) == 1:
                action = action[0]
            if isinstance(action, (np.floating, float)):
                action = int(round(action))

            action = np.array([action])

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward.sum()
        episode_length += 1

        if terminated.any() or truncated.any():
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            obs, _ = env.reset()

    env.close()

    eval_time = time.perf_counter() - start_time

    return EvalResult(
        mean_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        std_reward=float(np.std(episode_rewards)) if episode_rewards else 0.0,
        mean_length=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        episodes=len(episode_rewards),
        eval_time=eval_time,
    )


def evaluate_submission(
    submission_dir: Path,
    env_name: str = "cartpole",
    num_episodes: int = 100,
    seed: int = 42,
    timeout: float = 60.0,
) -> EvalResult:
    """
    Evaluate a student submission directory.

    Expected structure:
        submission_dir/
        ├── policy.py      # defines load_policy(checkpoint_path) -> callable
        └── checkpoint.pt  # model weights
    """
    submission_dir = Path(submission_dir)
    policy_path = submission_dir / "policy.py"
    checkpoint_path = submission_dir / "checkpoint.pt"

    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy.py in {submission_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint.pt in {submission_dir}")

    policy = load_policy_from_submission(policy_path, checkpoint_path)
    return evaluate_policy(policy, env_name, num_episodes, seed, timeout)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a student submission")
    parser.add_argument("submission_dir", type=Path, help="Path to submission directory")
    parser.add_argument("--env", default="cartpole", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout in seconds")

    args = parser.parse_args()

    result = evaluate_submission(
        args.submission_dir,
        env_name=args.env,
        num_episodes=args.episodes,
        seed=args.seed,
        timeout=args.timeout,
    )

    print(f"Evaluation Results:")
    print(f"  Episodes completed: {result.episodes}")
    print(f"  Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
    print(f"  Mean episode length: {result.mean_length:.2f}")
    print(f"  Eval time: {result.eval_time:.2f}s")
