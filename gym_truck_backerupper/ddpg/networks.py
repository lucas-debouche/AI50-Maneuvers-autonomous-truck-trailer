"""Actor and Critic networks for DDPG (PyTorch)."""
import torch
import torch.nn as nn


# This function allows us to build same MLP architecture for both Actor and Critic, with different input/output sizes.
# Exemple : _mlp([15, 256, 256, 1]) builds a network with input size 15, two hidden layers of size 256, and output size 1.
# Code : nn.Sequential(nn.Linear(15, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)) -> _mlp([15, 256, 256, 1])
def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _small_last_layer(net, bound=3e-3):
    """DDPG paper trick: tiny final weights -> initial outputs close to 0
    (Q ~ 0, steering ~ 0), which avoids saturating tanh at the start."""
    last = [m for m in net if isinstance(m, nn.Linear)][-1]
    nn.init.uniform_(last.weight, -bound, bound)
    nn.init.uniform_(last.bias, -bound, bound)


class Actor(nn.Module):
    """mu(s): state (15) -> steering in [-1, 1]."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, act_dim])
        _small_last_layer(self.net)

    def forward(self, s):
        return torch.tanh(self.net(s))  # tanh => valid action in [-1, 1]


class Critic(nn.Module):
    """Q(s, a): (state, action) -> expected discounted return."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp([obs_dim + act_dim, *hidden, 1])
        _small_last_layer(self.net)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)) # concatenate state and action along the last dimension, then pass through the network to get Q-value
