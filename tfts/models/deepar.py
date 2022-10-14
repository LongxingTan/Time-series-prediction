"""
`DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
<https://arxiv.org/abs/1704.04110>`_
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout

from tfts.layers.deepar_layer import GaussianLayer

params = {
    "rnn_size": 64,
    "dense_size": 16,
}


class DeepAR(object):
    def __init__(self, custom_model_params={}):
        """DeepAR Network

        :param custom_model_params:
        """
        self.params = params
        cell = tf.keras.layers.GRUCell(units=self.params["rnn_size"])
        self.rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.bn = BatchNormalization()
        self.dense = Dense(units=self.params["dense_size"], activation="relu")
        self.gauss = GaussianLayer(units=1)

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x, _ = self.rnn(x)
        x = self.dense(x)
        loc, scale = self.gauss(x)
        return loc, scale
