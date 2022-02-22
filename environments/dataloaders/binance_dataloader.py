from os import stat_result
from random import random, randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.control_flow_ops import group
from tqdm import tqdm

class BinanceDataLoader():
    def __init__(self, path, nbins=30, sample_size = 64, train_steps = 1000, *args, **kwargs):
        self.path = path
        self.df = pd.read_csv(path, usecols=['date', 'close', 'Volume BTC', 'unix']).dropna()
        self.df = self.df.reindex(index=self.df.index[::-1]).reset_index()
        self.step = 1

        self.nbins = nbins
        self.change_nbins = 3

        #self.seed = randint(100, len(self.df) - (train_steps * 1.1) * self.step)

        # Calculate steps, skip if consequent prices are same
        self.steps = train_steps #(len(self.df) - self.df['close'].diff().eq(0).sum() - sample_size - self.seed) // self.step 

        self.sample_size = sample_size
    
    def process_data(self, train = False):
        self.seed = randint(100, len(self.df) - (self.steps * 1.1) * self.step)
        
        # Minute Dataset
        df = self.df
        ss = self.sample_size

        while True:
            i = self.seed
            train_i = 0
            while train_i <= self.steps + 1:

                data = df.iloc[i - ss: i]
                
                
                # Hourly Dataset with window size ws1
                prices = np.array(data['close'])
                volume = np.array(data['Volume BTC'])

                price_old = prices[-1]
                price_curr = df.iloc[i + self.step]['close']

                if train:
                    i = randint(self.sample_size + 1, len(self.df) - self.step - 1)
                else:
                    i += self.step

                if price_old == price_curr:
                    continue

                try:
                    volume_n = (volume - volume.min()) / (volume.max() - volume.min())
                    prices_n = (prices - prices.min()) / (prices.max() - prices.min())
                except ValueError:
                    tqdm.write('Value Error!')
                    continue

                price_old_n = prices_n[-1]
                price_curr_n = (price_curr - prices.min()) / (prices.max() - prices.min())
                

                obs = np.concatenate([prices_n, volume_n])
                                
                # plt.plot(obs, label='Price WS = 10m')
                # # plt.plot(x1, label='Volume WS = 10m')
                # # plt.plot(x2, label='Price WS = 30m')
                # # plt.plot(x3, label='Volume WS = 30m')
                # # plt.plot(x4, label='Price WS = 1m')
                # # plt.plot(x5, label='Volume WS = 1m')
                # plt.legend()
                # plt.show()
                
                train_i += 1
                yield {'observation': np.array(obs), 'price': price_curr, 'price_old': price_old, 'price_n': price_curr_n, 'price_old_n': price_old_n}
            
            yield False

            tqdm.write('Data reset!')
        
            # Calculate steps, skip if consequent prices are same
            

    def get_steps(self):
        return self.steps


    def fibonacci_retracement(self, d_h, d_l, ):
        levels = np.array([0.236, 0.382,  0.5, 0.618, 0.786])
        f = []
        for i in range(1, 13):
            hs = d_h.tail(i)
            ls = d_l.tail(i)
            m = hs.max()
            l = ls.min()

            x = l + (m - l) * levels
            f.append(x)
 
        return np.array(f)          
            
    def date_weights(self, n):
        x = np.linspace(0, 1, n)
        f_x = -x**4 - x**3 + 2*x**2 + x
        try:
            return f_x / f_x.max()
        except Exception:
            return np.ones(n)

    def get_histogram(self, data, weights=True):
        hist = np.zeros(self.nbins)
        bins = np.linspace(data.min(), data.max(), self.nbins)

        if weights:
            w = self.date_weights(data.shape[0])
        else:
            w = np.ones(data.shape[0])

        i = 0
        for _, d in data.items():
            x = np.digitize(d, bins)
            hist[x - 1] += w[i]
            i += 1

        return hist


    def load_data(self):
        return pd.read_csv(self.path)

    def get_df(self):
        return self.df

    def get_data(self):
        return self.train_data, self.val_data

    def get_train_size(self):
        return len(self.train)

    def get_val_size(self):
        return len(self.val)

# dl = DataLoader('data/Binance_BTCUSDT_1h.csv')
# df = dl.get_df().tail(3000)
# f = dl.fibonacci_retracement(df['high'], df['low'])
