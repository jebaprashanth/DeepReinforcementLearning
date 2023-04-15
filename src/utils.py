import pandas as pd
import numpy as np
import torch

"""
This code defines several functions for preprocessing financial data 
and generating noise for use in a portfolio optimization task.
"""
# ----------------------------------------------------------------------------------

"""
preprocess_data(file_path, window_length) takes a CSV file of financial data 
and a window length, and returns a PyTorch tensor of overlapping windows of data. 
It first loads the data using pd.read_csv, then normalizes the data 
by subtracting the mean and dividing by the standard deviation. 
It then creates a list of overlapping windows of length window_length 
by iterating over the data and appending each window to the list. 
Finally, it converts the list of windows to a PyTorch tensor and returns it.

"""
def preprocess_data(file_path, window_length):
    # Load the historical financial data
    data = pd.read_csv(file_path)

    # Normalize the data
    data = (data - data.mean()) / data.std()

    # Create overlapping windows of data
    data_windows = []
    for i in range(len(data) - window_length):
        data_window = data[i:i+window_length]
        data_windows.append(data_window.values)

    # Convert the data to PyTorch tensors
    data_windows = torch.tensor(data_windows, dtype=torch.float32)

    return data_windows


"""
generate_noise(mu, sigma, size) generates random noise from a normal distribution 
with mean mu, standard deviation sigma, and size size using torch.normal.
"""
def generate_noise(mu, sigma, size):
    return torch.normal(mu, sigma, size=size)


"""
calculate_portfolio_weights(actions, available_funds) takes a tensor of actions 
representing portfolio weights and a scalar available_funds, and returns 
a tensor of portfolio allocations. It first applies a softmax function 
to the actions using torch.softmax to convert them to probabilities 
that sum to 1. It then multiplies the resulting probabilities 
by available_funds to obtain the allocation for each asset, 
and returns a tensor of these allocations.
"""
def calculate_portfolio_weights(actions, available_funds):
    # Softmax function to convert actions to portfolio weights
    weights = torch.softmax(actions, dim=-1)

    # Calculate portfolio allocations using available funds
    allocations = weights * available_funds

    return allocations
