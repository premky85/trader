from re import S
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
from models.LSTMNet import LSTMNet

def fibonacci_retracement(d_h, d_l):
        levels = np.array([0.236, 0.382,  0.5, 0.618, 0.786], dtype='float32')
        #f = []
        i = 8
        #for i in range(1, 13):
        hs = d_h.tail(i)
        ls = d_l.tail(i)
        m = hs.max()
        l = ls.min()

        x = l + (m - l) * levels
        #f.append(x)
 
        return x#np.array(f)          

class TraderEnv(Env):
    def __init__(self, *args, **kwargs):
        # Load data
        self.data = pd.read_csv('/share/Disk_2/Projects/trader/data/Binance_BTCUSDT_1h.csv')
        # action > 0.001: buy
        # -0.001 < action < 0.001: hold
        # action < -0.001: sell
        self.action_space = Box(low=np.array([-0.01]), high=np.array([0.01]))
        # Balance array
        self.observation_space = Box(low=np.array([0, -1, -1, 0, 0, 0, 0, 0]) - 0.001, high=np.array([1, 1, 1, 1, 1, 1, 1, 1]) + 0.001)
        # Start balance
        self.balance_currency = 1000
        self.balance_stock = 0
        # How long do we trade
        self.trading_time = 1000
        # Data index 
        self.index = 100
        # Load model for heuristics
        self.window_size = 80
        self.prediction_model = LSTMNet((self.window_size, 1), 1, weights='/share/Disk_2/Projects/trader/weights/binance-1h/LSTMNet-18_08_2021-21:04:04/LSTMNet_binance-1h_0078_mae_0.0337.h5').get_model()
        # Init state
        df = self.data.iloc[self.index - self.window_size:self.index]
        prices = df['close'].to_numpy()
        current_price = self.data.iloc[self.index]['close']

        highs  = df['high']
        lows = df['low']
        fibonaccis = fibonacci_retracement(highs, lows)

        # Normalize data
        try:
            norm_prices = (prices - prices.min()) / (prices.max() - prices.min())
            fibonaccis = (fibonaccis - prices.min()) / (prices.max() - prices.min())
        except ValueError:
            print('ValueError')

        model_input = np.array([np.expand_dims(norm_prices, axis=1)])
        predicted = self.prediction_model.predict(model_input).item(0) #* (prices.max() - prices.min()) + prices.min()

        self.total_balance = self.balance_currency + self.balance_stock * current_price

        self.state = np.concatenate([np.array([predicted, self.balance_currency / 1e6, self.balance_stock / 1e6], dtype='float32'), fibonaccis])

        

    def step(self, action):
        # Select rows data for prediction
        self.index += 1
        df = self.data.iloc[self.index - + self.window_size:self.index]
        prices = df['close']
        current_price = self.data.iloc[self.index]['close']

        highs  = df['high']
        lows = df['low']
        fibonaccis = fibonacci_retracement(highs, lows)

        # Normalize data
        try:
            norm_prices = (prices - prices.min()) / (prices.max() - prices.min())
            fibonaccis = (fibonaccis - prices.min()) / (prices.max() - prices.min())
        except ValueError:
            print('ValueError')

        model_input = np.array([np.expand_dims(norm_prices, axis=1)])
        predicted = self.prediction_model.predict(model_input).item(0) #* (prices.max() - prices.min()) + prices.min()

        if action <= 0.0001 and action >= -0.0001:
            None

        else:
            self.balance_currency -= current_price * action[0]
            self.balance_stock += action[0]
        

        self.trading_time -= 1

        current_balance = self.balance_currency + self.balance_stock * current_price
        reward = current_balance - self.total_balance
        self.total_balance = current_balance

        self.state = np.concatenate([np.array([predicted, self.balance_currency / 1e6, self.balance_stock / 1e6], dtype='float32'), fibonaccis])


        return self.state, reward.item(0), self.trading_time <= 0, {}





    def reset(self):
        # Start balance
        self.balance_currency = 1000
        self.balance_stock = 0
        # How long do we trade
        self.trading_time = 1000
        # Data index 
        self.index = 100
        # Load model for heuristics
        self.window_size = 80
        #self.prediction_model = LSTMNet((self.window_size, 1), 1, weights='/share/Disk_2/Projects/trader/weights/binance-1h/LSTMNet-18_08_2021-21:04:04/LSTMNet_binance-1h_0078_mae_0.0337.h5').get_model()
        # Init state
        df = self.data.iloc[self.index - + self.window_size:self.index]
        prices = df['close'].to_numpy()

        highs  = df['high']
        lows = df['low']
        fibonaccis = fibonacci_retracement(highs, lows)

        # Normalize data
        try:
            norm_prices = (prices - prices.min()) / (prices.max() - prices.min())
            fibonaccis = (fibonaccis - prices.min()) / (prices.max() - prices.min())
        except ValueError:
            print('ValueError')

        model_input = np.array([np.expand_dims(norm_prices, axis=1)])
        predicted = self.prediction_model.predict(model_input).item(0) #* (prices.max() - prices.min()) + prices.min()

        self.state = np.concatenate([np.array([predicted, self.balance_currency / 1e6, self.balance_stock / 1e6], dtype='float32'), fibonaccis])

        return self.state




