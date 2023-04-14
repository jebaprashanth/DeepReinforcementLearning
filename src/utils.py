import pandas as pd
import numpy as np
import torch

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

def generate_noise(mu, sigma, size):
    return torch.normal(mu, sigma, size=size)

def calculate_portfolio_weights(actions, available_funds):
    # Softmax function to convert actions to portfolio weights
    weights = torch.softmax(actions, dim=-1)

    # Calculate portfolio allocations using available funds
    allocations = weights * available_funds

    return allocations
