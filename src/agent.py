import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from models import Actor, Critic
from replay_buffer import ReplayBuffer
from ou_noise import OUActionNoise



class PortfolioManager:
    def __init__(self, state_dim, action_dim, max_action, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor and critic models
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Initialize target actor and critic models
        self.target_actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Set up action noise
        self.noise = OUActionNoise(mu=np.zeros(action_dim))

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(args.buffer_size)

        # Set up hyperparameters
        self.discount = args.discount
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.batch_size = args.batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = action + self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=128):
        # Sample batches from replay buffer
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute Q targets
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch)
            next_q_values = self.target_critic(next_state_batch, next_actions)
            q_targets = reward_batch + (1 - done_batch) * self.discount * next_q_values

        # Compute critic loss
        q_values = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(q_values, q_targets)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actions).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target actor and critic networks
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

