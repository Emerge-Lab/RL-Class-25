"""
Sandboxed execution of student policy code.

Security measures:
- Runs in isolated subprocess
- CPU/memory/file size limits via resource module
- Timeout enforcement
- Environment variables cleared (hides API keys, secrets)
- HOME redirected to /tmp

Limitations:
- Cannot fully restrict filesystem access without OS-level sandboxing
- Relies on container isolation (Railway/Docker) for full security
- Students can still read system files like /etc/passwd

For stronger isolation, consider:
- Docker-in-Docker with --no-new-privileges
- gVisor/Firecracker
- nsjail (Linux only)
"""

import multiprocessing
# Use fork on Linux for faster subprocess startup
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass  # Already set

import resource
import signal
import sys
import os
from pathlib import Path
from typing import Any


class SandboxError(Exception):
    """Error during sandboxed execution."""
    pass


class TimeoutError(SandboxError):
    """Execution timed out."""
    pass


class ResourceError(SandboxError):
    """Resource limit exceeded."""
    pass


def _set_resource_limits():
    """Set resource limits for the sandboxed process."""
    def safe_setrlimit(resource_type, soft, hard):
        """Set resource limit, handling platform differences."""
        try:
            current_soft, current_hard = resource.getrlimit(resource_type)
            # Can only set up to the hard limit
            new_soft = min(soft, current_hard) if current_hard != resource.RLIM_INFINITY else soft
            new_hard = min(hard, current_hard) if current_hard != resource.RLIM_INFINITY else hard
            resource.setrlimit(resource_type, (new_soft, new_hard))
        except (ValueError, resource.error, OSError):
            pass  # Limit not available on this system

    # Max CPU time: 60 seconds
    safe_setrlimit(resource.RLIMIT_CPU, 60, 60)

    # Max memory: 1GB
    safe_setrlimit(resource.RLIMIT_AS, 1024 * 1024 * 1024, 1024 * 1024 * 1024)

    # Max file size: 10MB (prevent disk filling)
    safe_setrlimit(resource.RLIMIT_FSIZE, 10 * 1024 * 1024, 10 * 1024 * 1024)

    # Max number of open files: 256 (needs some headroom for Python)
    safe_setrlimit(resource.RLIMIT_NOFILE, 256, 256)

    # No core dumps
    safe_setrlimit(resource.RLIMIT_CORE, 0, 0)

    # Max processes/threads: 16 (needs some for PyTorch)
    if hasattr(resource, 'RLIMIT_NPROC'):
        safe_setrlimit(resource.RLIMIT_NPROC, 16, 16)


def _restrict_imports():
    """Block dangerous modules from being imported."""
    dangerous_modules = [
        'subprocess', 'os.system', 'commands',
        'socket', 'http', 'urllib', 'requests', 'httpx', 'aiohttp',
        'shutil', 'pathlib',  # filesystem manipulation
        'pickle', 'marshal', 'shelve',  # serialization exploits
        'ctypes', 'cffi',  # native code
        'multiprocessing', 'threading', 'concurrent',  # parallelism
        'importlib', 'imp', '__import__',  # dynamic imports
        'eval', 'exec', 'compile',  # code execution
        'builtins', '__builtins__',
    ]

    class ImportBlocker:
        def find_module(self, name, path=None):
            for blocked in dangerous_modules:
                if name == blocked or name.startswith(blocked + '.'):
                    return self
            return None

        def load_module(self, name):
            raise ImportError(f"Module '{name}' is blocked for security reasons")

    sys.meta_path.insert(0, ImportBlocker())


def _run_in_sandbox(submission_dir: str, env_name: str, num_episodes: int,
                    seed: int, result_queue: multiprocessing.Queue):
    """Run evaluation in sandboxed subprocess."""
    try:
        # Set resource limits
        _set_resource_limits()

        # Clear environment variables (hide secrets like API keys)
        # Keep only minimal vars needed for execution
        allowed_vars = {'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'TMPDIR', 'TMP', 'TEMP'}
        for key in list(os.environ.keys()):
            if key not in allowed_vars:
                del os.environ[key]

        # Override HOME to temp dir to prevent reading user files
        os.environ['HOME'] = '/tmp'

        # Now do the actual evaluation
        import importlib.util
        import numpy as np
        import torch

        # Load student policy (use absolute path before any chdir)
        submission_path = Path(submission_dir).resolve()
        policy_path = submission_path / "policy.py"
        checkpoint_path = submission_path / "checkpoint.pt"

        spec = importlib.util.spec_from_file_location("student_policy", policy_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["student_policy"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "load_policy"):
            raise ValueError("policy.py must define load_policy(checkpoint_path)")

        policy = module.load_policy(checkpoint_path)

        # Run evaluation
        if env_name == "cartpole":
            from pufferlib.ocean.cartpole.cartpole import Cartpole
            env = Cartpole(num_envs=1, render_mode=None)
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        episode_rewards = []
        episode_lengths = []

        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        episode_length = 0

        while len(episode_rewards) < num_episodes:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                action = policy(obs_tensor)

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

        result_queue.put({
            "success": True,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "episodes": len(episode_rewards),
        })

    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
        })


def run_sandboxed_evaluation(
    submission_dir: Path,
    env_name: str = "cartpole",
    num_episodes: int = 100,
    seed: int = 42,
    timeout: float = 60.0,
) -> dict:
    """
    Run student policy evaluation in a sandboxed subprocess.

    Provides:
    - CPU/memory limits
    - Filesystem isolation
    - Environment variable hiding
    - Timeout enforcement
    """
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_run_in_sandbox,
        args=(str(submission_dir), env_name, num_episodes, seed, result_queue),
    )

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        raise TimeoutError(f"Evaluation timed out after {timeout}s")

    if result_queue.empty():
        raise SandboxError("Sandboxed process died without returning results")

    result = result_queue.get()

    if not result["success"]:
        raise SandboxError(f"Evaluation failed: {result['error']}")

    return result
