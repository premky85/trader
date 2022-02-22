import tensorflow as tf
import numpy as np
from random import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_loader import OUActionNoise

model = tf.keras.models.load_model('./checkpoints/ActorNet')
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

df = pd.read_csv('./data/Binance_BTCUSDT_minute.csv', usecols=['date', 'close', 'Volume USDT']).dropna()
df = df.reindex(index=df.index[::-1])

trading_summary = {'bank': [], 'USD': [], 'Profit': []}


ws = 100
ss = 20

bank = 0
bank_threshold = 1e4

max_balance_0 = 1000
min_balance_0 = 0
min_balance_net = 0

max_balance_1 = 1e-5
min_balance_1 = 1e-30


balance_0 = 1000         # money balance
balance_1 = 1e-5         # crypto balance
balance_net = 1000       # net worth

max_percentage_0 = 0.5   # maximum single transaction buy
max_percentage_1 = 0.3   # maximum single transaction sell

step = 60
steps = len(df) - ws

pbar = tqdm(total=steps)


for i, row in df.iterrows():
    if i % step != 0:
        continue

    d = df[(df['date'] < row['date'])]
    d = d.sort_values(by=['date'])
    if len(d) < ws:
        continue
        
    
    
    d = d.tail(ws)

    prices = d['close']
    volume = d['Volume USDT']

    try:
        volume_n = (volume - volume.min()) / (volume.max() - volume.min())
        prices_n = (prices - prices.min()) / (prices.max() - prices.min())
        me = np.mean(prices)
    except ValueError:
        continue

    prices_n = prices_n.tail(ss + 1).head(ss)
    volume_n = volume_n.tail(ss + 1).head(ss)

    obs = np.concatenate([prices_n, volume_n]).flatten()
    state = np.array([np.concatenate([obs, [balance_0 / max_balance_0, balance_1 / max_balance_1]])])
    price = np.array(prices)[-1]
    price_old = np.array(prices)[-2]

    action = tf.squeeze(model(state)).numpy()
    noise = ou_noise()
    action = np.squeeze(np.clip(action + noise, -1.0, 1.0))


    # buy or hold
    if action >= 0 : 
        if balance_0 > max_balance_0 / 5:
            transaction_ammount = action * max_percentage_0 * balance_0
            balance_0 -= transaction_ammount
            balance_1 += transaction_ammount / price_old

    
    else:                       # sell
        transaction_ammount = action * max_percentage_1 * balance_1
        balance_1 += transaction_ammount 
        balance_0 -= transaction_ammount * price_old
        

    new_net = price * balance_1 + balance_0
    reward = new_net - balance_net
    balance_net = new_net

    if balance_0 >= bank_threshold:
        bank += 2000
        balance_0 -= 2000
        new_net = price * balance_1 + balance_0

    profit = balance_net + bank - 1000
    trading_summary['bank'].append(bank)
    trading_summary['Profit'].append(profit)
    trading_summary['USD'].append(balance_0)
    pbar.set_postfix({'Total profit': profit, 'USD': balance_0, 'BTC': balance_1}, refresh=False)
    pbar.update(step)


plt.plot(trading_summary['USD'], label='USD')
plt.plot(trading_summary['bank'], label='Bank')
plt.plot(trading_summary['Profit'], label='Profit')
plt.legend()
plt.show()




