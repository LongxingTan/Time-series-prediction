# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-05
# paper:
# Other implementation: https://github.com/arrigonialberto86/deepar

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tfts.layers.deepar_layer import GaussianLayer

params = {
    'rnn_size': 64,
    'dense_size': 16,
}


class DeepAR(object):
    def __init__(self, custom_model_params={}):
        """ DeepAR Network
        reference paper: `DeepAR: Probabilistic forecasting with autoregressive recurrent networks`
        :param custom_model_params:
        """
        self.params = params
        cell = tf.keras.layers.GRUCell(units=self.params['rnn_size'])
        self.rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.bn = BatchNormalization()
        self.dense = Dense(units=self.params['dense_size'], activation='relu')
        self.gauss = GaussianLayer(units=1)

    def __call__(self, x):
        x, _ = self.rnn(x)
        x = self.dense(x)
        loc, scale = self.gauss(x)
        return loc, scale
