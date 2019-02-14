from prepare_model_input import Input_builder
from models.seq2seq import Time_Seq2seq
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

class Config:
    input_seq_length=5
    output_seq_length=5
    hidden_size=[8]
    learning_rate=10e-3
    n_epochs=15
    batch_size=1
    n_layers=1
config = Config()


def run_prediction():
    config = Config()


    dataset = pd.read_csv('../data/LSTM_data.csv', usecols=[1], engine='python', sep=',')
    dataset = dataset.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))

    input_builder = Input_builder()
    trainX, trainY = input_builder.create_seq2seq_input(input_seq_length=config.input_seq_length,
                                                        output_seq_length=config.output_seq_length)
    testX, testY = input_builder.create_seq2seq_input(input_seq_length=config.input_seq_length,
                                                      output_seq_length=config.output_seq_length)

    seq2seq = Time_Seq2seq(config)
    seq2seq.train(trainX, trainY)