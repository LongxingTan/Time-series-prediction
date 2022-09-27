#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, GRU
from tfts.layers.attention_layer import FullAttention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding, TokenRnnEmbedding
from tfts.models.transformer import Encoder


params = {
    'n_encoder_layers': 1,
    'use_token_embedding': False,
    'attention_hidden_sizes': 32 * 1,
    'num_heads': 2,
    'attention_dropout': 0.,
    'ffn_hidden_sizes': 32 * 1,
    'ffn_filter_sizes': 32 * 1,
    'ffn_dropout': 0.,
    'layer_postprocess_dropout': 0.,
    'scheduler_sampling': 1,  # 0 means teacher forcing, 1 means use last prediction
    'skip_connect': False
}


class Bert(object):
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        print(params)

        self.encoder_embedding = TokenEmbedding(
            params['attention_hidden_sizes'])  # DataEmbedding(params['attention_hidden_sizes'])
        # self.spatial_drop = SpatialDropout1D(0.1)
        # self.tcn = ConvTemporal(kernel_size=2, filters=32, dilation_rate=6)
        self.encoder = Encoder(
            params['n_encoder_layers'],
            params['attention_hidden_sizes'],
            params['num_heads'],
            params['attention_dropout'],
            params['ffn_hidden_sizes'],
            params['ffn_filter_sizes'],
            params['ffn_dropout'])

        self.project1 = Dense(predict_sequence_length, activation=None)
        # self.project1 = Dense(48, activation=None)

        # self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation='relu')

        # self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation='relu')

        # self.forecasting = Forecasting(predict_sequence_length, self.params)
        # self.pool1 = AveragePooling1D(pool_size=6)
        # self.rnn1 = GRU(units=1, activation='tanh', return_state=False, return_sequences=True, dropout=0)
        # self.rnn2 = GRU(units=32, activation='tanh', return_state=False, return_sequences=True, dropout=0)
        # self.project2 = Dense(48, activation=None)
        # self.project3 = Dense(1, activation=None)
        #
        # self.dense_se = Dense(16, activation='relu')
        # self.dense_se2 = Dense(1, activation='sigmoid')

    def __call__(self, inputs, teacher=None):
        # inputs:
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, decoder_features = inputs
            # encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            decoder_features = None

        encoder_features = self.encoder_embedding(encoder_features)
        # encoder_features = self.spatial_drop(encoder_features)
        # encoder_features_res = self.tcn(encoder_features)
        # encoder_features += encoder_features_res

        memory = self.encoder(encoder_features, src_mask=None)  # batch * train_sequence * (hidden * heads)
        encoder_output = memory[:, -1]

        # encoder_output = self.bn1(encoder_output)
        encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.bn2(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)
        # outputs = tf.repeat(outputs, [6]*48, axis=1)

        # se = self.dense_se(decoder_features)  # batch * pred_len * 1
        # se = self.dense_se2(se)
        # outputs = tf.math.multiply(outputs, se)

        # memory
        # x2 = self.rnn1(memory)
        # outputs += 0.2 * x2

        # outputs2 = self.project2(encoder_output)  # 48
        # outputs2 = tf.repeat(outputs2, repeats=[6]*48, axis=1)
        # outputs2 = tf.expand_dims(outputs2, -1)
        # outputs += outputs2

        # outputs = self.forecasting(encoder_features, teacher)
        # outputs = tf.math.cumsum(outputs, axis=1)

        ## grafting
        base = decoder_features[:, :, -1:]
        outputs += base
        return outputs


class Forecasting(tf.keras.layers.Layer):
    def __init__(self, predict_sequence_length, params) -> None:
        super().__init__()
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = TokenEmbedding(params['attention_hidden_sizes'])
        self.encoder = Encoder(
            params['n_encoder_layers'],
            params['attention_hidden_sizes'],
            params['num_heads'],
            params['attention_dropout'],
            params['ffn_hidden_sizes'],
            params['ffn_filter_sizes'],
            params['ffn_dropout'])
        self.dense = Dense(1)

    def call(self, src, teacher=None, training=None):
        outputs = []
        for i in range(self.predict_sequence_length):
            # batch * seq * fea => batch * seq * fea
            inp = self.encoder_embedding(src)
            out = self.encoder(inp)
            out = out[:, -1:]
            out = self.dense(out)
            outputs.append(out)
            if training:
                p = np.random.uniform(low=0, high=1, size=1)[0]
                src = tf.concat([src[:, 1:], teacher[:, i: i + 1]], axis=1)
            else:
                src = tf.concat([src[:, 1:], out], axis=1)

        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.squeeze(outputs, 1)
        return outputs
