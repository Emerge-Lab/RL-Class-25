# Homework 2, Problem 4: DQN Training on CartPole

In this problem, you will put together all the components from Problem 2 into a working DQN training loop on CartPole-v1.

**Note:** DQN is often not a very performant algorithm compared to policy gradient methods like PPO. We have you implement it because its *components* (replay buffers, target networks, epsilon-greedy exploration, and TD learning) show up throughout modern RL.

## Setup

From the main repo directory:
```bash
uv sync
```

## Environment: CartPole-v1

Gymnasium's CartPole-v1:
- **Observations**: 4 floats (cart position, cart velocity, pole angle, pole angular velocity)
- **Actions**: Discrete(2) -- push left or push right
- **Rewards**: +1 per timestep the pole stays upright
- **Max episode length**: 500 steps

## Your Task

Implement the `train()` function in `train_dqn.py`. Everything else is provided:
- Network architecture (`QNetwork` from Problem 2)
- Hyperparameters (tuned values that work)
- Evaluation and plotting functions

Your `train()` function should:
1. Create the environment, online/target networks, optimizer, and `NStepReplayBuffer`
2. Run an epsilon-greedy step loop, pushing transitions to the buffer
3. After a warmup period, train on minibatches from the buffer
4. Periodically hard-update the target network
5. Return a list of episode rewards

## Components from Problem 2

Your training loop will use these functions from your Problem 2 implementation:
- `QNetwork` -- the Q-value network architecture
- `NStepReplayBuffer` -- stores transitions and computes n-step returns
- `batch_to_tensors` -- converts sampled transitions to tensors
- `epsilon_greedy_action` -- selects actions with exploration
- `linear_epsilon_decay` -- anneals epsilon over training
- `compute_double_dqn_target` -- computes TD targets with Double DQN
- `compute_td_loss` -- computes the Huber loss
- `hard_update` -- copies online network to target network

## Running

```bash
cd homeworks/homework_2/problem_4
uv run python train_dqn.py
```

You should see evaluation reward reach 500 (the maximum) within 200k timesteps. The learning curve is saved to `dqn_cartpole.png`.


## Leaderboard Submission

After training, submit your policy to the course leaderboard:

1. Go to https://eval-server-production-c3fe.up.railway.app/
2. Select **CartPole DQN (HW2 Problem 4)**
3. Upload your `policy.py` and `checkpoint.pt` files

Your `policy.py` must define a `load_policy(checkpoint_path)` function that returns a **callable**. The callable receives a `torch.FloatTensor` observation and should return an action (int, tensor, or numpy array). Example:

```python
def load_policy(checkpoint_path):
    model = QNetwork(state_dim=4, action_dim=2)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    def policy(obs):
        with torch.no_grad():
            q_values = model(obs.unsqueeze(0) if obs.dim() == 1 else obs)
            return q_values.argmax(dim=-1)

    return policy
```

Your `checkpoint.pt` should be the saved `model.state_dict()` from training.

The server evaluates your policy over 100 episodes. Higher mean reward is better (max 500).

## References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Deep RL with Double Q-learning (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)
