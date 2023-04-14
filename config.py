class Config:
    def __init__(self):
        # Paths
        self.data_path = "data/"
        self.models_path = "models/"

        # Hyperparameters
        self.gamma = 0.99
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.tau = 0.005
        self.buffer_size = 1000000
        self.batch_size = 128

        # Environment settings
        self.stock_data_filename = "stock_data.csv"
        self.economic_data_filename = "economic_data.csv"
        self.initial_capital = 1000000
        self.max_buy_pct = 0.05
        self.max_sell_pct = 0.05
        self.transaction_cost_pct = 0.001
        self.reward_scaling = 1e-4
        self.lookback_window = 50
        self.num_assets = 30
        self.trading_period = 30

        # Training settings
        self.num_episodes = 1000
        self.max_steps = 2000
        self.replay_buffer_size = 100000

        # Testing settings
        self.test_episodes = 10
        self.test_max_steps = 2000


