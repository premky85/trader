from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import filters
from sklearn import preprocessing


class BinanceDataLoader:
    def __init__(
        self,
        path,
        nbins: int = 30,
        sample_size: int = 64,
        train_steps: int = 1000,
        overfit=False,
        *args,
        **kwargs
    ):
        self.path = path
        self.df = pd.read_csv(
            path, usecols=["date", "close", "Volume USDT", "unix"]
        ).dropna()
        self.df = self.df.reindex(index=self.df.index[::-1]).reset_index()

        self.step = 24
        self.sample_multiplier = 3
        self.overfit = overfit

        self.nbins = nbins
        self.change_nbins = 3

        # self.seed = randint(100, len(self.df) - (train_steps * 1.1) * self.step)

        # Calculate steps, skip if consequent prices are same
        self.steps = train_steps  # (len(self.df) - self.df['close'].diff().eq(0).sum() - sample_size - self.seed) // self.step

        self.sample_size = sample_size

        self.sample_sizes = (
            sample_size,
            sample_size,
            sample_size,
            sample_size,
            sample_size,
            sample_size,
        )

        if overfit:
            self.df = self.df[5000:11000]

        self.scaler = preprocessing.StandardScaler()

    def process_data(self, validation=False):

        df = self.df
        ss = self.sample_size
        indicator_ss = ss

        while True:
            if self.overfit:
                i = 2000
            else:
                i = randint(
                    self.sample_size * (self.sample_multiplier + 1),
                    len(self.df) - (self.steps * 1.1) * self.step,
                )

            train_i = 0
            while True:
                done = train_i > self.steps + 1

                big_data = df.iloc[i - (ss * self.sample_multiplier) : i]
                # data = df.iloc[i - ss : i]

                # Hourly Dataset with window size ws1

                big_prices = np.array(big_data["close"])
                big_volume = np.array(big_data["Volume USDT"])

                prices = big_prices[-ss:]  # np.array(data["close"])
                volume = big_volume[-ss:]  # np.array(data["Volume BTC"])

                price_old = prices[-1]
                price_curr = df.iloc[i + self.step]["close"]

                p = preprocessing.scale(
                    np.concatenate([prices, [price_curr]]).astype(float)
                )

                # if train:
                #     i = randint(
                #         self.sample_size * (self.sample_multiplier + 1),
                #         len(self.df) - self.step - 1,
                #     )
                # else:
                #     i += self.step

                i += self.step  # if validation else 3  # self.step
                done = done or i > self.df.shape[0]

                if price_old == price_curr or volume[0] == 0 or prices[0] == 0:
                    continue

                # try:
                #     big_volume_n = (big_volume - big_volume.min()) / (
                #         big_volume.max() - big_volume.min()
                #     )  # volume / volume[0] - 1 #
                #     big_prices_n = (big_prices - big_prices.min()) / (
                #         big_prices.max() - big_prices.min()
                #     )  # prices / prices[0] - 1 #
                #     volume_n = big_volume_n[-ss:]
                #     prices_n = big_prices_n[-ss:]
                # except ValueError as ve:
                #     tqdm.write("Value Error!")
                #     continue

                # price_old_n = prices_n[-1]
                # price_curr_n = (price_curr - big_prices.min()) / (
                #     big_prices.max() - big_prices.min()
                # )  # price_curr / prices[0] - 1 #

                sma = BinanceDataLoader.sma(
                    a=big_prices,
                    output_size=indicator_ss,
                    window=self.sample_size // 4,
                )
                macd, macd_s, macd_h = BinanceDataLoader.macd(
                    a=big_prices,
                    output_size=indicator_ss,
                    macd_trigger_window=self.sample_size // 2,
                    window_s=self.sample_size,
                    window_l=self.sample_size * 2,
                )

                obs = list(
                    map(
                        lambda a: self.scaler.fit_transform(np.expand_dims(a, axis=1)),
                        [
                            prices,
                            volume,
                            sma,
                            macd,
                            macd_s,
                            macd_h,
                        ],
                    )
                )
                # obs = [np.array(a) for a in obs]
                obs = np.concatenate([np.squeeze(a) for a in obs])

                # TODO: standardize price for reward

                # plt.plot(prices_n, label='Normalized price')
                # plt.legend()
                # plt.show()

                train_i += 1
                yield {
                    "observation": np.array(obs, dtype=np.float32),
                    "price": price_curr,
                    "price_old": price_old,
                    "price_n": p[-1],
                    "price_old_n": p[-2],
                    "done": done,
                }

                if done:
                    break

            # Calculate steps, skip if consequent prices are same

    def get_sample_sizes(self):
        return self.sample_sizes

    @staticmethod
    def sma(a, output_size=64, window=60):
        output_size = int(output_size)
        assert len(a) > output_size + window

        ma = filters.uniform_filter1d(a, size=window)
        return ma[-output_size:]

    @staticmethod
    def ema(a, output_size=64, window=60):
        output_size = int(output_size)
        assert len(a) >= output_size + window
        if isinstance(a, (np.ndarray, list)):
            df = pd.Series(data=a)
        else:
            df = a

        a = df.ewm(span=window, adjust=False, min_periods=output_size).mean()

        if output_size > 0:
            a = a[-output_size:]
            assert len(a) == output_size

        return a

    @staticmethod
    def macd(a, output_size=64, window_s=500, window_l=1000, macd_trigger_window=300):
        output_size = int(output_size)
        assert window_l > window_s
        assert len(a) >= output_size + window_l

        fast = BinanceDataLoader.ema(a=a, output_size=0, window=window_s)
        slow = BinanceDataLoader.ema(a=a, output_size=0, window=window_l)
        macd = fast - slow

        macd_s = BinanceDataLoader.ema(
            a=macd, output_size=0, window=macd_trigger_window
        )

        macd_h = macd - macd_s

        return (
            np.array(macd)[-output_size:],
            np.array(macd_s)[-output_size:],
            np.array(macd_h)[-output_size:],
        )

    def get_steps(self):
        return self.steps

    def fibonacci_retracement(
        self,
        d_h,
        d_l,
    ):
        levels = np.array([0.236, 0.382, 0.5, 0.618, 0.786])
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
        f_x = -(x**4) - x**3 + 2 * x**2 + x
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
