"""
This implementation defines the Actor and Critic classes. 
The Actor takes in observations and outputs actions. 
It has a series of fully connected layers with ReLU activation, 
followed by a final output layer with a hyperbolic tangent activation 
to ensure that the output actions are within the range of -1 to 1. 
The Critic takes in observations and actions and outputs the state-action value function. 
It has a series of fully connected layers with ReLU activation, 
followed by a final output layer with a single output value. 
Note that the Critic class concatenates the observation and 
action inputs before passing them through the fully connected layers.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.output_layer(x))
        return x

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim + action_dim, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
