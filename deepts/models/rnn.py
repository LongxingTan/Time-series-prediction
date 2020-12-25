# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-05
# paper:
# Other implementation: https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation

params = {
    'rnn_size': 64,
    'dense_size': 16,
    'num_stacked_layers': 1,
}


class RNN(object):
    def __init__(self, custom_model_params={}):
        self.params = params
        cell = tf.keras.layers.GRUCell(units=self.params['rnn_size'])
        self.rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.bn = BatchNormalization()
        self.dense = Dense(units=self.params['dense_size'])

    def __call__(self, x):
        r, _ = self.rnn(x)
        r = self.bn(r)
        r = Dropout(0.25)(r)

        y = self.dense(r)
        y = Activation('relu')(y)
        return y

