"""
`Long short term memory
<http://www.bioinf.jku.at/publications/older/2604.pdf>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    AveragePooling1D,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    GRUCell,
    LSTMCell,
    TimeDistributed,
)

from tfts.layers.attention_layer import Attention

from .base import BaseConfig, BaseModel


class RNNConfig(BaseConfig):
    model_type = "rnn"

    def __init__(
        self,
        rnn_hidden_size=64,
        rnn_type="gru",
        bi_direction=False,
        dense_hidden_size=32,
        num_stacked_layers=1,
        scheduled_sampling=0,
        use_attention=False,
    ):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_type = rnn_type
        self.bi_direction = bi_direction
        self.dense_hidden_size = dense_hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.scheduled_sampling = scheduled_sampling
        self.use_attention = use_attention


class RNN(BaseModel):
    """tfts RNN model"""

    def __init__(self, config=None, predict_sequence_length: int = 1):
        super().__init__(config)
        if config is None:
            config = RNNConfig()
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(
            rnn_size=config.rnn_hidden_size, rnn_type=config.rnn_type, dense_size=config.dense_hidden_size
        )
        self.project1 = Dense(predict_sequence_length, activation=None)

        self.dense1 = Dense(128, activation="relu")
        self.bn = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense2 = Dense(128, activation="relu")
        self.drop2 = Dropout(0.25)

    def __call__(self, inputs, teacher=None, return_dict: Optional[bool] = None):
        """RNN model call

        Parameters
        ----------
        inputs : Union[list, tuple, dict, tf.Tensor]
            Input data.
        teacher : tf.Tensor, optional
            Teacher signal for training, by default None.
        return_dict : bool, optional
            Whether to return a dictionary, by default None.

        Returns
        -------
        tf.Tensor
            Model output.
        """
        x, encoder_feature = self._prepare_inputs(inputs)
        encoder_outputs, encoder_state = self.encoder(encoder_feature)

        if self.config.rnn_type == "lstm":
            concat_layer = tf.keras.layers.Concatenate(axis=-1)
            encoder_output = concat_layer(encoder_state)
        else:
            encoder_output = encoder_state

        encoder_output = self.dense1(encoder_output)
        encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)

        # outputs = tf.expand_dims(outputs, -1)
        expand_dims_layer = tf.keras.layers.Reshape((outputs.shape[1], 1))
        outputs = expand_dims_layer(outputs)

        return outputs

    def _prepare_inputs(self, inputs):
        """Prepare the inputs for the encoder.

        Parameters
        ----------
        inputs : Union[list, tuple, dict, tf.Tensor]
            Raw inputs.

        Returns
        -------
        tuple
            Prepared inputs.
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, _ = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            x = inputs
            encoder_feature = x
        return x, encoder_feature


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_size: int,
        rnn_type: str = "gru",
        rnn_dropout: float = 0,
        dense_size: int = 32,
        num_stacked_layers: int = 1,
        bi_direction: bool = False,
        return_state: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn_type = rnn_type.lower()
        self.rnn_size = rnn_size
        self.rnn_dropout = rnn_dropout
        self.dense_size = dense_size
        self.num_stacked_layers = num_stacked_layers
        self.bi_direction = bi_direction
        self.return_state = return_state

    def build(self, input_shape):
        self.rnns = []
        return_state = False if self.bi_direction else True

        for _ in range(self.num_stacked_layers):
            if self.rnn_type == "gru":
                rnn = GRU(
                    units=self.rnn_size,
                    activation="tanh",
                    return_state=return_state,
                    return_sequences=True,
                    dropout=self.rnn_dropout,
                )
            elif self.rnn_type == "lstm":
                rnn = LSTM(
                    units=self.rnn_size,
                    activation="tanh",
                    return_state=return_state,
                    return_sequences=True,
                    dropout=self.rnn_dropout,
                )
            else:
                raise ValueError(f"No supported RNN type: {self.rnn_type}")

            if self.bi_direction:
                rnn = Bidirectional(rnn)

            self.rnns.append(rnn)

        super(Encoder, self).build(input_shape)

    def call(self, inputs: tf.Tensor):
        """RNN encoder call

        Parameters
        ----------
        inputs : tf.Tensor
            3d time series input, (batch_size, sequence_length, num_features)

        Returns
        -------
        tf.Tensor or tuple
            If return_state is False:
                output tensor: (batch_size, sequence_length, rnn_size)
            If return_state is True:
                For GRU: (output, state)
                    output: (batch_size, sequence_length, rnn_size)
                    state: (batch_size, rnn_size)
                For LSTM: (output, state_h, state_c)
                    output: (batch_size, sequence_length, rnn_size)
                    state_h: (batch_size, rnn_size)
                    state_c: (batch_size, rnn_size)
        """
        x = inputs

        for i, layer in enumerate(self.rnns):
            is_last_layer = i == len(self.rnns) - 1

            if is_last_layer and self.return_state:
                if self.bi_direction:
                    if self.rnn_type == "gru":
                        x = layer(x)
                    else:  # lstm
                        x = layer(x)
                else:
                    if self.rnn_type == "gru":
                        x, state = layer(x)
                        return x, state
                    else:  # lstm
                        x, state_h, state_c = layer(x)
                        state = (state_h, state_c)
                        return x, state
            else:
                x = layer(x)

        return x

    def get_config(self):
        config = {
            "rnn_type": self.rnn_type,
            "rnn_size": self.rnn_size,
            "rnn_dropout": self.rnn_dropout,
            "dense_size": self.dense_size,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RNN2(object):
    def __init__(self, predict_sequence_length=3, config=RNNConfig()) -> None:
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.rnn = GRU(
            units=config.rnn_hidden_size, activation="tanh", return_state=True, return_sequences=True, dropout=0
        )
        self.dense1 = Dense(predict_sequence_length)
        self.dense2 = Dense(1)

    def __call__(self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None):
        """Another RNN model

        Parameters
        ----------
        inputs : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, _ = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs

        encoder_shape = tf.shape(encoder_features)
        future = tf.zeros([encoder_shape[0], self.predict_sequence_length, encoder_shape[2]])
        encoder_features = tf.concat([encoder_features, future], axis=1)
        output, state = self.rnn(encoder_features)
        output = self.dense2(output)

        return output[:, -self.predict_sequence_length :]
