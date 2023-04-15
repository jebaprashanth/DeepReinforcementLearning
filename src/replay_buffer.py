import numpy as np
import torch
"""
This code defines a class ReplayBuffer that implements 
a replay buffer used in reinforcement learning. 
The buffer stores experience tuples (state, action, reward, next state, done) 
generated during agent-environment interactions, 
which are later used for training the agent.
"""
class ReplayBuffer:

    """
    The class initializes an empty buffer with specified dimensions 
    obs_dim for the observation space, action_dim for the action space, 
    and buffer_size for the maximum buffer size. 
    The class then defines five numpy arrays for storing the experience tuples: 
    obs_buf for the observations, next_obs_buf for the next observations, 
    action_buf for the actions, reward_buf for the rewards, 
    and done_buf for the done signals.
    """
    def __init__(self, obs_dim, action_dim, buffer_size):
        self.obs_buf = np.zeros([buffer_size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([buffer_size, obs_dim], dtype=np.float32)
        self.action_buf = np.zeros([buffer_size, action_dim], dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size


    """
    The store method takes in an experience tuple 
    and stores it in the next available index of the buffer. 
    If the buffer is full, it overwrites the oldest tuple.
    """
    def store(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    """
    The sample_batch method randomly samples a batch 
    of experience tuples of size batch_size from the buffer, 
    and returns the batch as a dictionary of tensors with keys obs, 
    next_obs, action, reward, and done. The tensor values are 
    the corresponding arrays of the experience tuples in the buffer. 
    The torch.tensor function is used to convert the numpy arrays 
    to PyTorch tensors.
    """
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=torch.tensor(self.obs_buf[idxs], dtype=torch.float),
                     next_obs=torch.tensor(self.next_obs_buf[idxs], dtype=torch.float),
                     action=torch.tensor(self.action_buf[idxs], dtype=torch.float),
                     reward=torch.tensor(self.reward_buf[idxs], dtype=torch.float),
                     done=torch.tensor(self.done_buf[idxs], dtype=torch.float))
        return batch
