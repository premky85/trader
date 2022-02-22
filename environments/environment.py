from re import S
from environments.dataloaders.binance_dataloader import BinanceDataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

import gym
from gym import spaces

class TraderEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, path, sample_size = 64, train_steps = 1000):
        super(TraderEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(np.array([0]), np.array([1]), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(np.array([0 for i in range(sample_size * 2)]), np.array([1 for i in range (sample_size * 2)]), dtype=np.float32)
        self.loader = BinanceDataLoader(path, sample_size=sample_size, train_steps=train_steps)
        self.steps = self.loader.get_steps()
        self.path = path

    
    def step(self, action):
        data = next(self.data_iterator)
        obs = data['observation']
        action = (action[0] - 0.5) * 2
        info = {}
        self.i += 1
        self.trading_summary['price'].append(1000 * data['price_old'] / self.price_init)

        reward = action * (data['price_n'] - data['price_old_n'])

        old_net = self.balance_0 + data['price_old'] * self.balance_1

        
        # Warmup
        # if self.i < 12000:
        #     reward_factor = 0
        # else:
        #     reward_factor = 1


        # buy or hold
        if action >= 0:
            # reward *= (self.balance_0 / self.max_balance_0)
            transaction_ammount = action * self.balance_0 # * self.max_percentage_0 
            # Prevent trade if agent wants to spend less than one dollar
            if transaction_ammount > 10:
                self.balance_0 -= transaction_ammount
                self.balance_1 += transaction_ammount / data['price_old']

        # sell
        else:
            # reward *= (self.balance_1 / self.max_balance_1)
            transaction_ammount = action * self.balance_1 # * self.max_percentage_1
            # Prevent trade if agent wants to spend less than one dollar
            if transaction_ammount * data['price_old'] < -10:
                self.balance_1 += transaction_ammount 
                self.balance_0 -= transaction_ammount * data['price_old']

        # Uncomment to enable gradial warmup 
        # reward *= self.i / self.steps

        
        new_net = data['price'] * self.balance_1 + self.balance_0

        # reward = (new_net - old_net)


        # # Adjust rewards
        # if action >= 0:
        #     # Hold when price drops
        #     if reward < 0:
        #         reward *= 10
        #     # Hold when price rises
        #     else:
        #         reward *= 2
        # else:
        #     # Sell when price rises
        #     if reward < 0:
        #         reward *= 3
        #     # Sell when price drops
        #     else:
        #         reward *= 5


        # if self.balance_0 >= self.bank_threshold:
        #     self.bank += 2000
        #     self.balance_0 -= 2000
        #     new_net = data['price'] * self.balance_1 + self.balance_0

        self.balance_net = new_net

        if self.i >= self.steps:
            done = True
            #tqdm.write('Environment reset!, Net worth: {}'.format(self.balance_net + self.bank))
            # self.plot_episode()
        elif self.balance_net < 100:
            done = True
            info = 'error'
        else:
            done = False

        
        self.max_balance_0 = self.balance_0 if self.balance_0 > self.max_balance_0 else self.max_balance_0
        self.max_balance_1 = self.balance_1 if self.balance_1 > self.max_balance_1 else self.max_balance_1
        self.step_profit = reward

        # if self.i < 10000:
        #    reward *= (self.i / self.steps) * 50
        
        return obs, reward, done, info

    def reset(self):

        self.trading_summary = {'bank': [], 'USD': [], 'Profit': [], 'price': []}

        
        
        self.data_iterator = self.loader.process_data(train=True)
        self.bank = 0
        self.bank_threshold = 5e4

        self.freeze_steps = 0

        self.max_balance_0 = 1000
        self.min_balance_0 = 0
        self.min_balance_net = 0

        self.max_balance_1 = 1e-4
        self.min_balance_1 = 1e-40


        self.balance_0 = 1000         # money balance
        self.balance_1 = 1e-5         # crypto balance
        self.balance_net = 1000       # net worth

        self.max_percentage_0 = 0.95   # maximum single transaction buy
        self.max_percentage_1 = 0.95  # maximum single transaction sell

        self.i = 0

        self.step_profit = 0
        
        self.pbar = tqdm(total=self.steps)

        data = next(self.data_iterator)

        self.price_init = data['price_old']
        
        return data['observation']


    
    def render(self, mode='human', close=False):
        profit = self.balance_net + self.bank - 1000
        # self.trading_summary['bank'].append(self.bank)
        # self.trading_summary['Profit'].append(profit)
        # self.trading_summary['USD'].append(self.balance_0)

        self.pbar.set_postfix({'Total profit': profit, 'Step profit': self.step_profit, 'USD': self.balance_0, 'BTC': self.balance_1}, refresh=False)
        self.pbar.update(1)

    def plot_episode(self):
        plt.plot(self.trading_summary['Profit'], label='Profit')
        plt.plot(self.trading_summary['USD'], label='USD')
        plt.plot(self.trading_summary['bank'], label='Bank')
        plt.legend()
        plt.show()
        

        







