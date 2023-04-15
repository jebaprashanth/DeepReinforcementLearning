import torch
import numpy as np


"""
This code defines a class Environment that simulates 
an investment environment for an agent to learn from. 
The class takes in a data parameter which contains the 
historical stock price data. The initial_investment parameter 
determines the amount of initial investment available to the agent.
"""
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


    """
    The reset method initializes the simulation environment 
    by resetting the current step to zero, setting the stock owned to zero, 
    and setting the cash in hand to the initial investment amount. 
    It returns the observation of the environment as an array.
    """
    def reset(self):
        self.current_step = 0
        self.stock_owned = np.zeros(self.stock_price_history.shape[1])
        self.cash_in_hand = self.initial_investment
        return self._get_obs()


    """
    The step method takes in an action parameter 
    which is a value between -1 and 1. It calculates the reward 
    for the agent based on the difference between the current and 
    previous value of the portfolio. It then updates the current step, 
    stock owned, and cash in hand based on the action taken by the agent. 
    The method returns the observation of the environment, reward, 
    done (a boolean flag indicating whether the simulation has ended), 
    and information about the current value of the portfolio.
    """
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


    """
    The _get_obs method returns an array 
    containing the stock prices and the cash in hand.
    """
    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.stock_price_history.shape[1]] = self.stock_price_history[self.current_step]
        obs[-1] = self.cash_in_hand
        return obs



    """
    The _get_val method calculates the current value of the portfolio 
    based on the stock prices and the cash in hand.
    """
    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price_history[self.current_step]) + self.cash_in_hand
