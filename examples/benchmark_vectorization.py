"""Benchmark: C vectorization vs multiprocessing for PufferLib environments."""

import time
import numpy as np
import gymnasium as gym
import pufferlib
import pufferlib.vector


# Must be top-level for multiprocessing to pickle
def gym_cartpole_creator(buf=None, seed=0):
    env = gym.make("CartPole-v1")
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def benchmark_native_c(num_envs, num_steps):
    """Benchmark native C-vectorized CartPole."""
    from pufferlib.ocean.cartpole.cartpole import Cartpole

    env = Cartpole(num_envs=num_envs, render_mode=None)
    env.reset(seed=42)

    actions = np.random.randint(0, 2, (num_steps, num_envs))

    start = time.perf_counter()
    for step in range(num_steps):
        env.step(actions[step])
    elapsed = time.perf_counter() - start

    env.close()
    return elapsed


def benchmark_serial(num_envs, num_steps):
    """Benchmark serial (no parallelism) vectorized CartPole."""
    vec_env = pufferlib.vector.make(
        gym_cartpole_creator,
        num_envs=num_envs,
        backend=pufferlib.vector.Serial,
    )
    vec_env.reset()

    actions = np.random.randint(0, 2, (num_steps, num_envs))

    start = time.perf_counter()
    for step in range(num_steps):
        vec_env.step(actions[step])
    elapsed = time.perf_counter() - start

    vec_env.close()
    return elapsed


def benchmark_multiprocessing(num_envs, num_steps, num_workers):
    """Benchmark multiprocessing-vectorized CartPole."""
    vec_env = pufferlib.vector.make(
        gym_cartpole_creator,
        num_envs=num_envs,
        backend=pufferlib.vector.Multiprocessing,
        num_workers=num_workers,
    )
    vec_env.reset()

    actions = np.random.randint(0, 2, (num_steps, num_envs))

    start = time.perf_counter()
    for step in range(num_steps):
        vec_env.step(actions[step])
    elapsed = time.perf_counter() - start

    vec_env.close()
    return elapsed


def main():
    num_envs = 8
    num_steps = 1000
    num_workers = 4

    print(f"Benchmarking {num_envs} environments for {num_steps} steps each")
    print(f"Total steps: {num_envs * num_steps:,}\n")

    # Native C vectorization
    t_native = benchmark_native_c(num_envs, num_steps)
    sps_native = (num_envs * num_steps) / t_native
    print(f"Native C:        {t_native:.3f}s ({sps_native:,.0f} steps/sec)")

    # Serial backend
    t_serial = benchmark_serial(num_envs, num_steps)
    sps_serial = (num_envs * num_steps) / t_serial
    print(f"Serial:          {t_serial:.3f}s ({sps_serial:,.0f} steps/sec)")

    # Multiprocessing backend
    t_mp = benchmark_multiprocessing(num_envs, num_steps, num_workers)
    sps_mp = (num_envs * num_steps) / t_mp
    print(f"Multiprocessing: {t_mp:.3f}s ({sps_mp:,.0f} steps/sec)")

    print(f"\nNative C is {t_serial/t_native:.0f}x faster than Serial")
    print(f"Native C is {t_mp/t_native:.0f}x faster than Multiprocessing")


if __name__ == "__main__":
    main()
