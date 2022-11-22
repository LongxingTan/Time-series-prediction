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

from tfts.layers.attention_layer import FullAttention

params = {
    "rnn_type": "gru",
    "bi_direction": False,
    "rnn_size": 64,
    "dense_size": 32,
    "num_stacked_layers": 1,
    "scheduler_sampling": 0,
    "use_attention": False,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class RNN(object):
    """RNN model"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        custom_model_params: Optional[Dict[str, Any]] = None,
        custom_model_head: Optional[Callable] = None,
    ):
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(params["rnn_type"], params["rnn_size"], dense_size=params["dense_size"])
        self.project1 = Dense(predict_sequence_length, activation=None)

        self.dense1 = Dense(128, activation="relu")
        self.bn = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense2 = Dense(128, activation="relu")
        self.drop2 = Dropout(0.25)

    def __call__(self, inputs, teacher=None):
        """_summary_

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
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs

        encoder_outputs, encoder_state = self.encoder(encoder_feature)
        # outputs = self.dense1(encoder_state)  # batch * predict_sequence_length
        # outputs = self.dense2(encoder_outputs)[:, -self.predict_sequence_length]
        if self.params["rnn_type"] == "lstm":
            encoder_output = tf.concat(encoder_state, axis=-1)
        else:
            encoder_output = encoder_state

        # encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        # encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            outputs = outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x, axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            outputs = outputs + x_mean
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_type, rnn_size, rnn_dropout=0, dense_size=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.rnn_dropout = rnn_dropout
        self.dense_size = dense_size

    def build(self, input_shape):
        if self.rnn_type.lower() == "gru":
            self.rnn = GRU(
                units=self.rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=self.rnn_dropout,
            )
        elif self.rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                units=self.rnn_size,
                activation="tanh",
                return_state=False,
                return_sequences=True,
                dropout=self.rnn_dropout,
            )
            self.rnn = Bidirectional(self.rnn)
            self.rnn2 = LSTM(
                units=self.rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=self.rnn_dropout,
            )
            # self.rnn2 = Bidirectional(self.rnn2)

        # self.dense1 = Dense(self.dense_size, activation='relu')
        # self.bn = BatchNormalization()
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        """RNN encoder call

        Parameters
        ----------
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            output of encoder, batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        """
        # inputs = self.bn(inputs)
        if self.rnn_type.lower() == "gru":
            output, state = self.rnn(inputs)  # state is equal to outputs[:, -1]
        elif self.rnn_type.lower() == "lstm":
            output = self.rnn(inputs)
            output, state_memory, state_carry = self.rnn2(output)
            state = (state_memory, state_carry)
        # encoder_hidden_state = tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        # output = self.dense1(output)  # => batch_size * input_seq_length * dense_size
        return output, state

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
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.rnn = GRU(units=params["rnn_size"], activation="tanh", return_state=True, return_sequences=True, dropout=0)
        self.dense1 = Dense(predict_sequence_length)
        self.dense2 = Dense(1)

    def __call__(self, inputs, teacher=None):
        """_summary_

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
