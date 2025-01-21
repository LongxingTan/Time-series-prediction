"""
`DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
<https://arxiv.org/abs/1704.04110>`_
"""

from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense

from tfts.tasks.auto_task import GaussianHead

from .base import BaseConfig, BaseModel


class DeepARConfig(BaseConfig):
    model_type: str = "deepar"

    def __init__(
        self,
        rnn_hidden_size: int = 64,
    ):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size


config: Dict[str, Any] = {
    "rnn_size": 64,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class DeepAR(BaseModel):
    """DeepAR Network"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        config=DeepARConfig(),
    ):

        super(DeepAR, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        cell = tf.keras.layers.GRUCell(units=self.config.rnn_hidden_size)
        self.rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.bn = BatchNormalization()
        self.dense = Dense(units=predict_sequence_length, activation="relu")
        self.gauss = GaussianHead(units=1)

    def __call__(
        self, inputs: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        """DeepAR

        Parameters
        ----------
        inputs : tf.Tensor
            3D input tensor for time series

        Returns
        -------
        distribution of prediction
            _description_
        """
        x, _ = self.rnn(inputs)
        x = self.dense(x)
        loc, scale = self.gauss(x)
        return loc, scale
