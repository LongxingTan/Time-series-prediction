"""
`DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
<https://arxiv.org/abs/1704.04110>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout

from tfts.layers.deepar_layer import GaussianLayer

from .base import BaseConfig, BaseModel


class DeepARConfig(BaseConfig):
    model_type = "deepar"

    def __init__(
        self,
        rnn_hidden_size=64,
    ):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size


config: Dict[str, Any] = {
    "rnn_size": 64,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class DeepAR(BaseModel):
    def __init__(
        self,
        predict_sequence_length: int = 1,
        config=DeepARConfig,
    ):
        """DeepAR Network"""
        super(DeepAR, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        cell = tf.keras.layers.GRUCell(units=self.config.rnn_hidden_size)
        self.rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.bn = BatchNormalization()
        self.dense = Dense(units=predict_sequence_length, activation="relu")
        self.gauss = GaussianLayer(units=1)

    def __call__(self, inputs: tf.Tensor, return_dict: Optional[bool] = None):
        """DeepAR

        Parameters
        ----------
        x : tf.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x, _ = self.rnn(inputs)
        x = self.dense(x)
        loc, scale = self.gauss(x)
        return loc, scale
