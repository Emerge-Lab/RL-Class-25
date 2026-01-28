# Problem 1: Tabular Q-Iteration for MountainCar

In this problem, you will implement Q-iteration (dynamic programming) for the MountainCar environment. The continuous observation space has been discretized into a 200x200 grid.

## Environment

**MountainCar** has a car stuck in a valley that must build momentum to reach the goal on the right.

**Observation space (2 dimensions):**
- Position: range [-1.2, 0.6], goal at position >= 0.5
- Velocity: range [-0.07, 0.07]

**Actions:**
- 0: Push left
- 1: No push
- 2: Push right

**Reward:** -1 for each timestep (encourages reaching the goal quickly)

## Transition Tables

We provide precomputed transition tables that describe the discretized environment dynamics.

> **WARNING:** Do NOT modify or regenerate the transition tables (`.npy` files). The grading server uses the same tables - changing them will cause your submission to fail.

### State Space Discretization

The state space is discretized into a 200x200 grid:

- **s0 (first index):** Position index, ranging from 0 to 199
  - Index 0 = position -1.2 (leftmost)
  - Index 199 = position 0.6 (rightmost)
  - Goal region (position >= 0.5) is roughly indices 189-199

- **s1 (second index):** Velocity index, ranging from 0 to 199
  - Index 0 = velocity -0.07 (moving left fastest)
  - Index 199 = velocity 0.07 (moving right fastest)
  - Index ~100 = velocity ~0 (stationary)

### Table Formats

**P (transition_next_states.npy):** shape `(200, 200, 3, 2)`

```python
P[s0, s1, a] = [s0', s1']  # numpy array of 2 integers
```

Given current state indices (s0, s1) and action a, returns the next state indices.

Example:
```python
next_state = P[100, 100, 2]  # State (100,100), action 2 (push right)
s0_next, s1_next = next_state[0], next_state[1]
# Now you can look up Q[s0_next, s1_next, :] to get Q-values at next state
```

**R (transition_rewards.npy):** shape `(200, 200, 3)`

```python
R[s0, s1, a] = reward  # single float, always -1.0
```

The immediate reward for taking action a in state (s0, s1).

**D (transition_dones.npy):** shape `(200, 200, 3)`

```python
D[s0, s1, a] = done  # boolean: True or False
```

Whether the episode terminates after taking action a. True only when the car reaches the goal.

## Your Task

### 1. Implement `q_iteration()` in `problem_1.py`

Implement Q-iteration using the Bellman optimality equation:

```
Q(s, a) = R(s, a) + gamma * (1 - D(s, a)) * max_a' Q(s', a')
```

where s' = P(s, a) is the next state.

**Algorithm:**
1. Initialize Q-table to zeros: shape (200, 200, 3)
2. Repeat until convergence or max_iterations:
   - For each state-action pair (s0, s1, a):
     - Look up next state: `s0', s1' = P[s0, s1, a]`
     - Look up reward: `r = R[s0, s1, a]`
     - Look up done: `d = D[s0, s1, a]`
     - Update: `Q_new[s0, s1, a] = r + gamma * (1 - d) * max_a' Q[s0', s1', a']`
   - If max|Q_new - Q| < theta: converged, stop
   - Q = Q_new
3. Return the converged Q-table

### 2. Implement `forward()` in `policy.py`

Implement the action selection method:
1. Discretize the observation using `self.discretize_state(obs)`
2. Look up the Q-values for that state in `self.q_table`
3. Return the action with the highest Q-value

## Running Your Solution

```bash
python problem_1.py
```

This will:
1. Load the transition tables
2. Run your Q-iteration implementation
3. Save the Q-table to `checkpoint.pt`
4. Evaluate your policy

## Submission

Submit two files to the leaderboard:
- `checkpoint.pt` - Your Q-table saved with torch.save (shape: 200x200x3)
- `policy.py` - With your implementation of the `forward()` method

## Grading

Your policy must achieve a **mean reward better than -150** to pass.

A well-implemented solution should:
- Consistently reach the goal (position >= 0.5)
- Complete episodes in fewer than 150 steps on average
- Achieve 100% success rate
