import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is PyTorch code for defining two neural network models 
for an actor-critic reinforcement learning algorithm.

Both models use PyTorch's nn.Module class as a parent class, 
which enables the models to be saved, loaded, and trained using 
PyTorch's built-in functionality. 
The nn.ModuleList class is used to create a list of layers 
that can be easily iterated over in the forward pass.
"""
# --------------------------------------------------------------------------------------

"""
The Actor model takes as input the observation state (obs_dim) 
and outputs an action (action_dim) to take in the environment. 
It has two hidden layers of size 256 each by default, 
with the number of hidden layers and their sizes 
specified by the hidden_sizes argument. 
The forward method performs a feedforward pass through the layers, 
applying a ReLU activation function to the hidden layers 
and a hyperbolic tangent activation function to the output layer.
"""
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



"""
The Critic model takes as input the observation state 
and the action taken in that state, and outputs an estimate of the value 
of that state-action pair. It also has two hidden layers 
of size 256 each by default, and takes in the same obs_dim 
and action_dim arguments as the Actor model. 
The forward method concatenates the observation and action inputs, 
passes them through the hidden layers with ReLU activation functions, 
and outputs a single scalar value with the output layer.
"""
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
