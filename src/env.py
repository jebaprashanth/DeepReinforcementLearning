"""
This implementation defines an Environment class that takes in 
a dataset of historical stock prices and an initial investment amount, 
and allows an agent to interact with the environment by taking actions 
and receiving rewards. The reset() method resets the environment to its 
initial state, while the step() method takes in an action and 
returns the next observation, reward, done flag, and info dictionary. 
The _get_obs() method returns the current state of the environment, 
which includes the current stock prices and the amount of cash the agent has on hand. 
The _get_val() method calculates the total value of the agent's portfolio at the current step.
"""

import torch
import numpy as np

class Environment:
    def __init__(self, data, initial_investment=20000):
        self.stock_price_history = data
        self.n_step = self.stock_price_history.shape[0]
        self.initial_investment = initial_investment
        self.current_step = None
        self.stock_owned = None
        self.cash_in_hand = None
        self.action_space = np.arange(-1, 1.1, 0.1)
        self.action_space_size = len(self.action_space)
        self.state_dim = self.stock_price_history.shape[1] + 1

    def reset(self):
        self.current_step = 0
        self.stock_owned = np.zeros(self.stock_price_history.shape[1])
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()
        self.current_step += 1

        if action >= 0:
            sales = self.stock_owned * self.stock_price_history[self.current_step]
            sales_cost = sales * 0.0015
            proceeds = np.sum(sales - sales_cost)
            buy_num_shares = np.floor((self.cash_in_hand - sales_cost) / self.stock_price_history[self.current_step])
            buy_cost = buy_num_shares * self.stock_price_history[self.current_step]
            buy_cost += buy_cost * 0.0015
            self.stock_owned += buy_num_shares
            self.cash_in_hand -= buy_cost
        else:
            sell_num_shares = np.floor((-action * self.stock_owned).clip(max=self.stock_owned))
            sales = sell_num_shares * self.stock_price_history[self.current_step]
            sales_cost = sales * 0.0015
            proceeds = np.sum(sales - sales_cost)
            self.stock_owned -= sell_num_shares
            self.cash_in_hand += proceeds

        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.current_step == self.n_step - 1
        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.stock_price_history.shape[1]] = self.stock_price_history[self.current_step]
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price_history[self.current_step]) + self.cash_in_hand
