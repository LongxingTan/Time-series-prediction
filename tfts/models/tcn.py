#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper:
# other implementations: https://github.com/philipperemy/keras-tcn
#                        https://github.com/locuslab/TCN
#                        https://github.com/emreaksan/stcn


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
from ..layers.wavenet_layer import Dense3D, ConvTime


params = {
    'dilation_rates': [2 ** i for i in range(4)],
    'kernel_sizes': [2 for i in range(4)],
    'filters': 128,
    'dense_hidden_size': 64
    'skip_connect': False,
}


class TCN(object):
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        print(params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(params['kernel_sizes'], params['dilation_rates'], params['filters'],
                               params['dense_hidden_size'])
        # self.dense2 = Dense(1)
        # self.dense3 = TimeDistributed(Dense(1))
        # self.pool = AveragePooling1D(pool_size=144, strides=144, padding='valid')

        self.project1 = Dense(predict_sequence_length, activation=None)
        # self.project1 = Dense(48, activation=None)

        # self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation='relu')

        # self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation='relu')

    def __call__(self, inputs, teacher=None):
        # inputs:
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, decoder_features = inputs
            # encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            decoder_features = None

        # encoder_features = self.pool(encoder_features)  # batch * n_train_days * n_feature

        encoder_outputs, encoder_state = self.encoder(encoder_features)
        # outputs = self.dense1(encoder_state)  # batch * predict_sequence_length
        # outputs = self.dense2(encoder_outputs)[:, -self.predict_sequence_length]  # if train_sequence > predict_sequence
        # print(len(encoder_outputs), encoder_outputs[0].shape, encoder_state.shape)

        memory = encoder_state[:, -1]
        encoder_output = self.drop1(memory)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.bn2(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)

        # outputs = tf.tile(outputs, (1, self.predict_sequence_length, 1))   # stupid
        # outputs = self.dense3(encoder_outputs)

        if self.params['skip_connect']:
            x_mean = tf.tile(tf.reduce_mean(x, axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            # x_mean = tf.tile(x, (1, 1, 1))  # 2 is predict_window/train_window
            outputs = outputs + x_mean
        return outputs


class Encoder(object):
    def __init__(self, kernel_sizes, dilation_rates, filters, dense_hidden_size):
        self.filters = filters
        self.conv_times = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilation_rates)):
            self.conv_times.append(ConvTemporal(filters=2 * filters,
                                                kernel_size=kernel_size,
                                                causal=True,
                                                dilation_rate=dilation))
        self.dense_time1 = Dense3D(units=filters, activation='tanh', name='encoder_dense_time1')
        self.dense_time2 = Dense3D(units=filters + filters, name='encoder_dense_time2')
        self.dense_time3 = Dense3D(units=dense_hidden_size, activation='relu', name='encoder_dense_time3')
        self.dense_time4 = Dense3D(units=1, name='encoder_dense_time_4')

    def forward(self, x):
        """
        :param x:
        :return: conv_inputs [batch_size, time_sequence_length, filters] * time_sequence_length
        """
        inputs = self.dense_time1(inputs=x)  # batch_size * time_sequence_length * filters

        skip_outputs = []
        conv_inputs = [inputs]
        for conv_time in self.conv_times:
            dilated_conv = conv_time(inputs)
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
            outputs = self.dense_time2(inputs=dilated_conv)
            skips, residuals = tf.split(outputs, [self.filters, self.filters], axis=2)
            inputs += residuals
            conv_inputs.append(inputs)  # batch_size * time_sequence_length * filters
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = self.dense_time3(skip_outputs)
        # y_hat = self.dense_time4(h)
        return conv_inputs[:-1], h

    def __call__(self, x):
        return self.forward(x)
