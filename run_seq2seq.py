from prepare_model_input import Input_builder
from models.seq2seq import Time_Seq2seq
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

#https://github.com/tensorflow/tensorflow/issues/4281
class Config(object):
    input_seq_length=5
    output_seq_length=5
    input_dim=1
    output_dim=1
    lstm_hidden_size=16
    num_stacked_layers=2
    lambda_l2_reg=0.0

    learning_rate = 10e-4
    n_epochs = 10
    batch_size = 1


def run_prediction_basic():
    config = Config()
    input_builder = Input_builder('LSTM_data.csv')
    trainX, trainY = input_builder.create_seq2seq_basic_input(input_seq_length=config.input_seq_length,
                                                        output_seq_length=config.output_seq_length)
    testX, testY = input_builder.create_seq2seq_basic_input(input_seq_length=config.input_seq_length,
                                                      output_seq_length=config.output_seq_length)

    sess = tf.Session()

    seq2seq = Time_Seq2seq(sess=sess,config=config)
    seq2seq.train(trainX, trainY)


if __name__ == '__main__':
    run_prediction_basic()