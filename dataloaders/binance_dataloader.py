from os import stat_result
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.python.ops.control_flow_ops import group

class BinanceDataLoader():
    def __init__(self, path, nbins=30, window_size=100, predict_size=5, *args, **kwargs):
        self.path = path
        self.df = self.load_data()
        self.df = self.df.dropna()
        self.nbins = nbins
        self.change_nbins = 3
        self.window_size = window_size
        self.predict_size = predict_size


        self.train, self.val = self.df.head(int(self.df.shape[0] * 0.8)), self.df.tail(int(self.df.shape[0] * 0.2))#train_test_split(self.df, test_size=0.2, random_state=42)

        self.train_data = self.process_data(self.train)
        self.val_data = self.process_data(self.val)

    
    def process_data(self, data):
        df = self.df
        while True:
            for i, row in data.iterrows():
                    
                d = df[(df['date'] < row['date'])]
                d = d.sort_values(by=['date'])

                open_prices = d['open'].tail(self.window_size + self.predict_size)
                close_prices = d['close'].tail(self.window_size + self.predict_size)


                norm_prices = close_prices#((close_prices - open_prices) / close_prices).to_numpy()
                try:
                    norm_prices = (norm_prices - norm_prices.min()) / (norm_prices.max() - norm_prices.min())
                    me = np.mean(norm_prices)
                except ValueError:
                    continue

                #hist = np.histogram(norm_prices, bins=self.change_nbins)
                past_prices, future_prices = norm_prices[:self.window_size], norm_prices[self.window_size:]
                #past_class = np.digitize(past_prices, np.concatenate([[-np.inf], np.histogram(past_prices, bins=self.change_nbins)[1], [np.inf]]))
                #future_class = np.digitize(future_prices, np.concatenate([[-np.inf], np.histogram(future_prices, bins=self.change_nbins)[1], [np.inf]]))

                


                inputs = np.array([np.expand_dims(past_prices, axis=1)])#np.array([np.concatenate((np.concatenate([o_h, c_h, h_h, l_h, t_h]).reshape(1, 5 * self.nbins), f), axis=None)])
                outputs = np.array([future_prices])
                #outputs = np.eye(6)[outputs]

#np.array([np.eye(self.change_nbins + 2)[chg_class - 1]]).astype('int32')

                if inputs.shape != (1, self.window_size,) and outputs.shape != (1, self.predict_size):
                    continue

                yield inputs, outputs 



            # plt.plot(o_h, label='open')
            # plt.plot(c_h, label='close')
            # plt.plot(h_h, label='high')
            # plt.plot(l_h, label='low')
            # plt.legend()
            # plt.show()


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
