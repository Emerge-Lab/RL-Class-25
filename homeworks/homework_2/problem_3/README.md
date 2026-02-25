# Homework 2, Problem 3: PPO Training on Pong

In this problem, you will put together all the components from Problem 1 into a working PPO training loop on PufferLib's native Pong environment.

## Setup

From the main repo directory:
```bash
uv sync
```

## Environment: PufferLib Pong

PufferLib's native Pong is a fast, lightweight environment:
- **Observations**: 8 floats (paddle/ball positions, normalized to [0, 1])
- **Actions**: Discrete(3) — stay, up, down
- **Rewards**: +1 for scoring, -1 for being scored on, +0.1 for bouncing the ball
- **Vectorized**: Runs `NUM_ENVS` environments in parallel via C code

## Your Task

Implement the `train()` function in `train_ppo.py`. Everything else is provided:
- Network architecture (`ActorCritic`)
- Hyperparameters (tuned values that work)
- Evaluation and plotting functions

Your `train()` function should:
1. Create the environment, model, optimizer, and `RolloutBuffer`
2. Collect rollouts by stepping the environment and storing transitions
3. Compute advantages with `buffer.compute_returns_and_advantages`
4. Run the PPO update loop (multiple epochs of minibatch updates)
5. Return a list of mean episode rewards per rollout

## Components from Problem 1

Your training loop will use these functions from your Problem 1 implementation:
- `RolloutBuffer` — stores trajectories from vectorized envs
- `sample_discrete_action` — samples actions during rollout collection
- `discrete_log_prob` — recomputes log probs during PPO updates
- `normalize_advantages` — normalizes advantages per minibatch
- `compute_policy_loss` — clipped PPO surrogate loss
- `compute_value_loss` — MSE value function loss
- `compute_entropy_bonus` — entropy bonus for exploration

## Running

```bash
cd homeworks/homework_2/problem_3
uv run python train_ppo.py
```

You should see positive average reward within 200k timesteps. The learning curve is saved to `ppo_pong.png`.

## Leaderboard Submission

After training, submit your policy to the course leaderboard:

1. Go to https://eval-server-production-c3fe.up.railway.app/
2. Select **Pong PPO (HW2 Problem 3)**
3. Upload your `policy.py` and `checkpoint.pt` files

Your `policy.py` must define a `load_policy(checkpoint_path)` function that returns a **callable**. The callable receives a `torch.FloatTensor` observation and should return an action (int, tensor, or numpy array). Example:

```python
def load_policy(checkpoint_path):
    model = ActorCritic(obs_dim=8, act_dim=3)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    def policy(obs):
        with torch.no_grad():
            logits, _ = model(obs.unsqueeze(0) if obs.dim() == 1 else obs)
            return logits.argmax(dim=-1)

    return policy
```

Your `checkpoint.pt` should be the saved `model.state_dict()` from training.

The server evaluates your policy over 100 episodes of Pong (max_score=5). Higher mean reward is better (max possible ~+5.9 with bounce bonuses).

## References

- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
