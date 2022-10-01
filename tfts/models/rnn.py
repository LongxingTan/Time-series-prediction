# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com

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
    "skip_connect": False,
}


class RNN(object):
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        print(params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(params["rnn_type"], params["rnn_size"], dense_size=params["dense_size"])
        self.project1 = Dense(predict_sequence_length, activation=None)

        self.dense1 = Dense(512, activation="relu")
        self.bn = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense2 = Dense(1024, activation="relu")
        self.drop2 = Dropout(0.25)
        # self.dense3 = TimeDistributed(Dense(1))
        # self.pool = AveragePooling1D(pool_size=144, strides=144, padding='valid')

    def __call__(self, inputs, teacher=None):
        # inputs:
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, _ = inputs
            # encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            # decoder_features = None

        # encoder_features = self.pool(encoder_features)  # batch * n_train_days * n_feature

        encoder_outputs, encoder_state = self.encoder(encoder_features)
        # outputs = self.dense1(encoder_state)  # batch * predict_sequence_length
        # outputs = self.dense2(encoder_outputs)[:, -self.predict_sequence_length]  # if train_sequence > predict_sequence
        if self.params["rnn_type"] == "lstm":
            encoder_output = tf.concat(encoder_state, axis=-1)
        else:
            encoder_output = encoder_state

        encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)

        if self.params["skip_connect"]:
            x_mean = tf.tile(tf.reduce_mean(x[:, -144:], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            # x_mean = tf.tile(x, (1, 1, 1))  # 2 is predict_window/train_window
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
        # outputs: batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
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


class RNNBaseline(object):
    # 仿照 官方base1: https://github.com/PaddlePaddle/PaddleSpatial/blob/main/apps/wpf_baseline_gru/model.py
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.rnn = GRU(units=params["rnn_size"], activation="tanh", return_state=True, return_sequences=True, dropout=0)
        self.dense1 = Dense(predict_sequence_length)
        self.dense2 = Dense(1)

    def __call__(self, inputs, teacher=None):
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, _ = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            # decoder_features = None

        encoder_shape = tf.shape(encoder_features)
        future = tf.zeros([encoder_shape[0], self.predict_sequence_length, encoder_shape[2]])
        encoder_features = tf.concat([encoder_features, future], axis=1)
        output, state = self.rnn(encoder_features)
        output = self.drop1(output)
        output = self.dense2(output)

        return output[:, -self.predict_sequence_length :, 0]


class RNNDay(object):
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.rnn = GRU(units=params["rnn_size"], activation="tanh", return_state=True, return_sequences=True, dropout=0)
        self.dense1 = Dense(predict_sequence_length)

    def __call__(self, inputs, teacher=None):
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, _ = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            # decoder_features = None

        # 计算按天的，采用avg_pool


class ESRNN(object):
    """ """

    def __init__(self) -> None:
        pass
