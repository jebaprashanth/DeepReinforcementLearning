"""
This implementation defines a ReplayBuffer class that stores 
transitions of observations, actions, rewards, next observations, and done flags. 
The store() method adds a new transition to the buffer, 
while the sample_batch() method returns a batch of 
transitions randomly sampled from the buffer. 
The implementation uses NumPy arrays to store the buffer, 
and converts them to PyTorch tensors when returning a batch.
"""

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, buffer_size):
        self.obs_buf = np.zeros([buffer_size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([buffer_size, obs_dim], dtype=np.float32)
        self.action_buf = np.zeros([buffer_size, action_dim], dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def store(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=torch.tensor(self.obs_buf[idxs], dtype=torch.float),
                     next_obs=torch.tensor(self.next_obs_buf[idxs], dtype=torch.float),
                     action=torch.tensor(self.action_buf[idxs], dtype=torch.float),
                     reward=torch.tensor(self.reward_buf[idxs], dtype=torch.float),
                     done=torch.tensor(self.done_buf[idxs], dtype=torch.float))
        return batch
