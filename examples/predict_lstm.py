import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_loader import *
from sklearn.preprocessing import scale


path = 'data/'
dataset = 'binance-1h'

p = path + 'Binance_BTCUSDT_1h.csv'

window_size = 80
predict_size = 20

model = LSTMNet((window_size, 1), 1, weights='weights/binance-1h/LSTMNet-18_08_2021-21:04:04/LSTMNet_binance-1h_0078_mae_0.0337.h5')
lstmnet = model.get_model()

dataframe = pd.read_csv(p)


for step in range(window_size + predict_size, dataframe.shape[0], window_size + predict_size):
    predicted = []
    df = dataframe.iloc[step:step + window_size + predict_size]
    close_prices = df['close'].head(window_size + predict_size)

    norm_prices = close_prices 

    try:
        norm_prices = (norm_prices - norm_prices.min()) / (norm_prices.max() - norm_prices.min())
        me = np.mean(norm_prices)
    except ValueError:
        print('ValueError')

    for i in range(predict_size):
        inputs = np.array([np.expand_dims(norm_prices[i:window_size + i], axis=1)])
        p = lstmnet.predict(inputs).item(0)
        predicted.append(p)

    l1 = np.linspace(0, window_size + predict_size - 1, window_size + predict_size)
    l2 = np.linspace(window_size-1, window_size + predict_size, predict_size)
    plt.plot(l1, norm_prices, label='Ground truth')
    plt.plot(l2, predicted, label='Predicted')
    plt.legend()

    plt.show()

