import torch
import numpy as np
from config import Config
from src.env import Environment
from src.agent import PortfolioManager
from src.replay_buffer import ReplayBuffer

# Set the device to use (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the configuration file
config = Config()

# Initialize the environment
env = Environment(config)

# Initialize the agent
agent = PortfolioManager(config, env.observation_space, env.action_space).to(device)

# Initialize the replay buffer
replay_buffer = ReplayBuffer(config.replay_buffer_size)

# Initialize the optimizer
actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.actor_lr)
critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=config.critic_lr)

# Start training
for episode in range(config.num_episodes):
    state = env.reset()

    for step in range(config.max_steps):
        # Get the action from the agent
        action = agent.act(state, device)

        # Take the action in the environment
        next_state, reward, done, info = env.step(action)

        # Add the transition to the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        # Update the agent
        if len(replay_buffer) > config.batch_size:
            experiences = replay_buffer.sample(config.batch_size)
            agent.update(experiences, actor_optimizer, critic_optimizer, device)

        # Update the state and total reward
        state = next_state

        if done:
            break

    # Print the results for this episode
    print(f"Episode {episode}: Total reward: {env.total_reward:.2f}")

# Test the agent
state = env.reset(test=True)

for step in range(config.max_steps):
    # Get the action from the agent
    action = agent.act(state, device, test=True)

    # Take the action in the environment
    next_state, reward, done, info = env.step(action)

    # Update the state and total reward
    state = next_state

    if done:
        break

# Print the final results
print(f"Final total reward: {env.total_reward:.2f}")
