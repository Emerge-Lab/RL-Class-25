# Homework 1: Value-Based Reinforcement Learning

This homework covers tabular Q-learning methods.

## Structure

- `Homework_1_theory.pdf` - Theory questions (submit on Gradescope)
- `problem_1/` - Tabular Q-iteration for MountainCar

## Submission Instructions

### Theory (Gradescope)
Submit your answers to the theory questions as a PDF on Brightspace.

### Programming Problems (Leaderboard)

For each programming problem, submit to the course leaderboard:

**Problem 1: MountainCar Q-Iteration**
- Submit: `policy.py` and `checkpoint.pt`
- Your policy must achieve mean reward > -150

### How to Submit to Leaderboard

1. Go to the course leaderboard: https://eval-server-production-c3fe.up.railway.app/
2. Select MountainCar
3. Upload your `policy.py` and `checkpoint.pt` files
4. Wait for evaluation results

### File Requirements

Your `policy.py` must contain:
- A `Policy` class with a `forward(obs)` method that returns an action
- A `load_policy(checkpoint_path)` function that returns a Policy instance

Your `checkpoint.pt` must be loadable by your `load_policy` function.

## Getting Started

First, follow the setup instructions in the main [README](../../README.md) to install dependencies using `uv sync`.

Then:
```bash
# Activate the virtual environment
source .venv/bin/activate

# Problem 1: Run Q-iteration
cd homeworks/homework_1/problem_1
python problem_1.py

# Test your policy locally
python policy.py
```

## Grading

- Theory questions: See Brightspace rubric
- Programming problems: Pass/fail based on leaderboard performance thresholds
