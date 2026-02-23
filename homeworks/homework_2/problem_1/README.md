# Homework 2, Problem 1: PPO Components

In this problem, you will implement the core components of the Proximal Policy Optimization (PPO) algorithm, based on the [CleanRL implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py).

## Setup

Make sure you have the environment set up (from the main repo directory):
```bash
uv sync
```

## Your Task

Implement all the functions in `ppo_components.py`. The functions are organized into four parts:

### Part 1: Core RL Computations
1. **`compute_returns`** - Compute discounted returns for a trajectory
2. **`compute_gae`** - Compute Generalized Advantage Estimation (GAE)
3. **`normalize_advantages`** - Normalize advantages to zero mean and unit variance

### Part 2a: Policy Distribution Functions
4. **`discrete_log_prob`** - Log probability for discrete (categorical) actions
5. **`discrete_entropy`** - Entropy of a categorical distribution
6. **`gaussian_log_prob`** - Log probability for continuous (Gaussian) actions
7. **`gaussian_entropy`** - Entropy of a diagonal Gaussian distribution

### Part 2b: Action Sampling
8. **`sample_discrete_action`** - Sample from categorical distribution with log probs
9. **`sample_continuous_action`** - Sample with tanh squashing for bounded actions
10. **`squashed_gaussian_log_prob`** - Log probability with Jacobian correction for tanh
11. **`clip_action`** - Clip continuous actions to a specified range

### Part 3: PPO Loss Functions
12. **`compute_policy_loss`** - Clipped PPO policy loss
13. **`compute_value_loss`** - Value function MSE loss
14. **`compute_entropy_bonus`** - Entropy bonus for exploration

### Part 4: Rollout Buffer
15. **`RolloutBuffer`** class - Buffer for storing trajectories from vectorized environments
    - `__init__` - Initialize storage tensors
    - `add` - Add a transition to the buffer
    - `compute_returns_and_advantages` - Compute GAE with proper bootstrapping
    - `get_batches` - Generate random minibatches for training

## Testing Your Implementation

Run the test suite to check your implementations:
```bash
uv run python test_components.py
```

The tests are organized by part and will show you which components pass and which fail.

## Tips

- Start with Part 1 (`compute_returns`) as it's the simplest
- `compute_gae` builds on similar concepts but uses TD errors
- For the policy distributions, use PyTorch's `log_softmax` for numerical stability
- The PPO loss needs to be **negated** because we maximize the objective but optimizers minimize
- The RolloutBuffer needs to handle partial episodes correctly using bootstrap values

## Key Concepts

### Discounted Returns
```
G_t = r_t + gamma * G_{t+1}
```

### GAE (Generalized Advantage Estimation)
```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  # TD error
A_t = delta_t + (gamma * lambda) * A_{t+1}   # GAE
```

### PPO Clipped Objective
```
ratio = pi(a|s) / pi_old(a|s)
L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```

### Vectorized Environments (Why We Use Them)

PPO is an **on-policy** algorithm: it collects data with the current policy, uses that data for a few gradient updates, then throws it away and collects fresh data. This means PPO needs a *lot* of environment interaction relative to the number of gradient steps. With a single environment, most of the wall-clock time is spent waiting for `env.step()` rather than training the network.

**Vectorized environments** solve this by running N copies of the environment in parallel. Instead of stepping one environment at a time, we step all N simultaneously, collecting N transitions per call. This gives us two key benefits:

1. **Throughput**: We collect data N times faster, which is critical for on-policy methods that can't reuse old data.
2. **Diversity**: Each environment runs an independent episode, so a single rollout of `num_steps` contains transitions from N different episodes at different stages. This provides more diverse training data per batch.

In practice, PPO implementations (like CleanRL) typically use 4-128 parallel environments to keep the GPU busy with training while CPUs handle environment stepping.

### Bootstrap Values (Why They Matter)

In standard RL with a single environment, you typically collect complete episodes before computing returns. But with vectorized environments (running N environments in parallel), you collect a fixed number of steps (`num_steps`) from ALL environments simultaneously, regardless of whether episodes have finished.

This creates a problem: when your rollout ends, some environments will be mid-episode:

```
Environment 0: [s0, s1, s2, s3, s4] → episode still running (no terminal state)
Environment 1: [s0, s1, DONE, s0', s1'] → episode ended, new one started
Environment 2: [s0, s1, s2, DONE, s0'] → episode ended at step 3
```

For Environment 0, we can't compute the true return because we don't know what future rewards will be. The solution is **bootstrapping**: we use our value function's estimate `V(s_last)` as a stand-in for all future rewards.

```python
# For an unfinished episode ending at state s_last:
G_t ≈ r_t + gamma * r_{t+1} + ... + gamma^k * V(s_last)
#                                            ↑
#                         Bootstrap: use value estimate instead of true future rewards
```

This is why `compute_returns_and_advantages` takes `last_value` and `last_done`:
- `last_value[i]` = V(s) for the state AFTER the last collected step in env i
- `last_done[i]` = 1 if env i's episode ended at the last step (don't bootstrap), 0 otherwise (do bootstrap)

Without bootstrapping, you'd either:
1. Throw away all partial episode data (wasteful)
2. Treat partial episodes as if they ended with 0 future reward (incorrect, biased)

Bootstrapping gives us an unbiased (in expectation) estimate that lets us use ALL collected data efficiently.

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GAE Paper](https://arxiv.org/abs/1506.02438)

## Acknowledgments

The testing framework is inspired by [Sasha Rush's Tensor Puzzles](https://github.com/srush/Tensor-Puzzles), an excellent resource for learning tensor operations through interactive puzzles.
