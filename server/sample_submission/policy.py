"""
Sample student policy submission.

This file must define: load_policy(checkpoint_path) -> callable
The callable must accept observations and return actions.
"""

import torch
import torch.nn as nn


class SimplePolicy(nn.Module):
    """A simple MLP policy for CartPole."""

    def __init__(self, obs_dim=4, hidden_dim=32, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        # obs shape: (batch, obs_dim) or (obs_dim,)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.net(obs)
        # Return greedy action
        return logits.argmax(dim=-1)


def load_policy(checkpoint_path):
    """Load the policy from a checkpoint file."""
    model = SimplePolicy()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return model
