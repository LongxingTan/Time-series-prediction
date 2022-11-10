"""
`DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
<https://arxiv.org/abs/1704.04110>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout

from tfts.layers.deepar_layer import GaussianLayer

params = {
    "rnn_size": 64,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class DeepAR(object):
    def __init__(
        self,
        predict_sequence_length: int = 1,
        custom_model_params: Optional[Dict[str, Any]] = None,
        custom_model_head: Optional[Callable] = None,
    ):
        """DeepAR Network

        :param custom_model_params:
        """
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length

        cell = tf.keras.layers.GRUCell(units=self.params["rnn_size"])
        self.rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.bn = BatchNormalization()
        self.dense = Dense(units=predict_sequence_length, activation="relu")
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
