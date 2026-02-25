# Homework 2: PPO and DQN

In this homework, you will implement the core components of two foundational deep RL algorithms — Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) — then train agents on Pong.

## Structure

The homework has four problems. In Problems 1 & 2 you implement individual components. In Problems 3 & 4 you wire those components into working training loops.

### Problem 1: PPO Components (`problem_1/`)
Implement the core building blocks of PPO:
- Discounted returns computation
- Generalized Advantage Estimation (GAE)
- Policy distribution functions (discrete and continuous)
- PPO loss functions (clipped policy loss, value loss, entropy bonus)
- Rollout buffer for vectorized environments

**Test your implementation:**
```bash
cd problem_1
uv run python test_components.py
```

### Problem 2: DQN Components (`problem_2/`)
Implement the core building blocks of DQN:
- Replay buffer for experience replay
- Epsilon-greedy exploration with linear decay
- TD target computation (standard and Double DQN)
- TD loss computation
- Target network updates (soft and hard)

**Test your implementation:**
```bash
cd problem_2
uv run python test_components.py
```

### Problem 3: PPO Training on Pong (`problem_3/`)
Put your PPO components together into a training loop on PufferLib's native Pong.

**Train your agent:**
```bash
cd problem_3
uv run python train_ppo.py
```

### Problem 4: DQN Training on CartPole (`problem_4/`)
Put your DQN components together into a training loop on CartPole-v1.

**Train your agent:**
```bash
cd problem_4
uv run python train_dqn.py
```

## Setup

From the main repo directory:
```bash
uv sync
```

## Expected Results

**Problem 1:** All 16 tests should pass. If you succeed, you'll get a surprise...!

**Problem 2:** All 24 tests should pass.

**Problem 3:** Your PPO agent should achieve positive average reward on Pong within 500k timesteps.

**Problem 4:** Your DQN agent should reach evaluation reward of 500 on CartPole within 200k timesteps.

## Leaderboard Submission

After training, submit your policies to the course leaderboard at https://eval-server-production-c3fe.up.railway.app/

- **Problem 3 (PPO Pong):** Select "Pong PPO (HW2 Problem 3)"
- **Problem 4 (DQN CartPole):** Select "CartPole DQN (HW2 Problem 4)"

For each submission, upload a `policy.py` and `checkpoint.pt`. Your `policy.py` must define a `load_policy(checkpoint_path)` function that returns a callable mapping observations to actions. See the individual problem READMEs for examples.

## References

**PPO:**
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [GAE Paper (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
- [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)

**DQN:**
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep RL (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Deep RL with Double Q-learning (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)

## Acknowledgments

The testing framework is inspired by [Sasha Rush's Tensor Puzzles](https://github.com/srush/Tensor-Puzzles).
