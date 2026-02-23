# Homework 2, Problem 2: DQN Components

In this problem, you will implement the core components of the Deep Q-Network (DQN) algorithm, based on the original DQN and Double DQN papers.

## Setup

Make sure you have the environment set up (from the main repo directory):
```bash
uv sync
```

## Your Task

Implement all the functions in `dqn_components.py`. The functions are organized into five parts:

### Part 1: Replay Buffer
1. **`ReplayBuffer`** class - Experience replay for breaking correlation between samples
    - `__init__` - Initialize the buffer
    - `push` - Add a transition
    - `sample` - Sample a random minibatch
    - `__len__` - Return buffer size
2. **`NStepReplayBuffer`** class - N-step replay buffer for improved credit assignment
    - `__init__` - Initialize with capacity, n_step, and gamma
    - `push` - Accumulate transitions and compute n-step returns
    - `_compute_nstep` - Compute discounted n-step return
3. **`batch_to_tensors`** - Convert a batch of transitions to PyTorch tensors

### Part 2: Epsilon-Greedy Exploration
4. **`epsilon_greedy_action`** - Select actions with epsilon-greedy exploration
5. **`linear_epsilon_decay`** - Compute epsilon on a linear decay schedule

### Part 3: TD Target Computation
6. **`compute_td_target`** - Standard DQN TD targets using max Q-value
7. **`compute_double_dqn_target`** - Double DQN targets that reduce overestimation bias

### Part 4: TD Loss Computation
8. **`compute_td_loss`** - Compute loss between predicted and target Q-values

### Part 5: Target Network Updates
9. **`soft_update`** - Polyak averaging of network parameters
10. **`hard_update`** - Full copy of network parameters

## Testing Your Implementation

Run the test suite to check your implementations:
```bash
uv run python test_components.py
```

The tests are organized by part and will show you which components pass and which fail.

## Tips

- Start with the Replay Buffer (Part 1) as it's the most straightforward
- For TD targets, pay careful attention to how terminal states are handled (the `done` flag)
- Double DQN's key idea: use one network to *pick* the best action and another to *evaluate* it
- The `gather()` function is essential for selecting Q-values of specific actions from a Q-value table
- Remember that TD targets should be detached from the computation graph

## Key Concepts

### Experience Replay
DQN stores transitions (s, a, r, s', done) in a replay buffer and trains on random minibatches. This breaks the temporal correlation between consecutive samples and stabilizes training.

### Epsilon-Greedy Exploration
```
With probability epsilon: take a random action
With probability 1-epsilon: take the greedy action (argmax Q)
```
Epsilon typically decays from ~1.0 to ~0.01 over training, shifting from exploration to exploitation.

### TD Targets (Standard DQN)
The Bellman equation gives us the target for updating Q-values:
```
y = r + gamma * max_a' Q(s', a') * (1 - done)
```
We minimize the error between our prediction Q(s, a) and this target.

### Double DQN (Why It Helps)
Standard DQN uses `max` to both select and evaluate actions, which causes overestimation bias (the max of noisy estimates is biased upward). Double DQN fixes this by decoupling selection and evaluation:
- **Online network** selects the best action: a* = argmax Q_online(s', a)
- **Target network** evaluates that action: Q_target(s', a*)

### N-Step Returns (Why They Help)
Standard DQN uses single-step TD updates: `y = r + gamma * max Q(s', a')`. This means reward information only propagates one step backward per update. If a good action's reward is 50 steps away, it takes ~50 updates for the Q-value to propagate back.

N-step returns fix this by looking ahead n steps:
```
R_n = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^{n-1} * r_{n-1}
y = R_n + gamma^n * max Q(s_n, a')
```

This propagates reward information n steps backward in a single update, dramatically improving credit assignment for environments with delayed rewards.

**Important**: When using n-step returns, the discount factor in the TD target becomes `gamma^n` instead of `gamma`, because we've already accounted for the first n steps of discounting in R_n.

### Target Networks (Why They're Needed)
In supervised learning, the targets are fixed. In DQN, the targets depend on the same network we're updating, creating a moving target problem. Target networks provide stable targets by using a delayed copy of the online network, updated either periodically (hard update) or gradually (soft/Polyak update).

## References

- [Playing Atari with Deep RL (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep RL (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Deep RL with Double Q-learning (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)

## Acknowledgments

The testing framework is inspired by [Sasha Rush's Tensor Puzzles](https://github.com/srush/Tensor-Puzzles), an excellent resource for learning tensor operations through interactive puzzles.
