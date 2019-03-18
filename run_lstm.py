from prepare_model_input import Input_builder
from models.LSTM import Time_LSTM
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class Config:
    n_states=5
    n_features=1

    n_layers = 1
    hidden_size=[128]
    learning_rate=10e-3
    n_epochs=15
    batch_size=1


def run_prediction():
    config = Config()

    '''
    data = ts.import_data('./Data/Repair_list.xlsx')
    # production_data = ts.import_data('Production_list.xlsx')
    calendar_data = ts.create_calendar_time(data)
    sample = calendar_data.loc[calendar_data['Fault location'] == '54027', ['Repair month', 'Value']][:-1]
    sample2 = sample.values

    x, y = ts.create_model_input(sample2, time_state=5)
    lstm = LSTM()
    lstm.train(x, y)
    lstm.predict_multi(x, predict_steps=20)


    dataset = pd.read_csv('../data/LSTM_data.csv', usecols=[1], engine='python', sep=',')
    dataset = dataset.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))
'''
    input_builder = Input_builder('./data/LSTM_data.csv')
    trainX, trainY = input_builder.create_RNN_input(time_state=config.n_states)
    testX, testY = input_builder.create_RNN_input(time_state=config.n_states)

    #session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session()

    lstm = Time_LSTM(sess=sess,config=config)
    lstm.train(trainX, trainY)
    trainPredict = lstm.predict_point(trainX)
    testPredict = lstm.predict_point(testX)
    sess.close()

'''
    trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
    testY = scaler.inverse_transform(testY)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[config.time_state:len(trainPredict) + config.time_state, :] = trainPredict
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (config.time_state * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
'''

if __name__ == '__main__':
    run_prediction()
