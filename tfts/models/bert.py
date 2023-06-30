"""
`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
<https://arxiv.org/abs/1810.04805>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    GRU,
    AveragePooling1D,
    BatchNormalization,
    Dense,
    Dropout,
    LayerNormalization,
    SpatialDropout1D,
)

from tfts.layers.attention_layer import FullAttention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding, TokenRnnEmbedding
from tfts.models.transformer import Encoder

params = {
    "n_encoder_layers": 1,
    "use_token_embedding": False,
    "attention_hidden_sizes": 32 * 1,
    "num_heads": 2,
    "attention_dropout": 0.0,
    "ffn_hidden_sizes": 32 * 1,
    "ffn_filter_sizes": 32 * 1,
    "ffn_dropout": 0.0,
    "layer_postprocess_dropout": 0.0,
    "scheduler_sampling": 1,  # 0 means teacher forcing, 1 means use last prediction
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class Bert(object):
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

        # DataEmbedding(params['attention_hidden_sizes'])
        self.encoder_embedding = TokenEmbedding(params["attention_hidden_sizes"])
        # self.spatial_drop = SpatialDropout1D(0.1)
        # self.tcn = ConvTemporal(kernel_size=2, filters=32, dilation_rate=6)
        self.encoder = Encoder(
            params["n_encoder_layers"],
            params["attention_hidden_sizes"],
            params["num_heads"],
            params["attention_dropout"],
            params["ffn_hidden_sizes"],
            params["ffn_filter_sizes"],
            params["ffn_dropout"],
        )

        self.project1 = Dense(predict_sequence_length, activation=None)
        # self.project1 = Dense(48, activation=None)

        # self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation="relu")

        # self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation="relu")

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
        """Bert model call

        Parameters
        ----------
        inputs : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None

        Returns
        -------
        tf.Tensor
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

        encoder_feature = self.encoder_embedding(encoder_feature)
        # encoder_features = self.spatial_drop(encoder_features)
        # encoder_features_res = self.tcn(encoder_features)
        # encoder_features += encoder_features_res

        memory = self.encoder(encoder_feature, mask=None)  # batch * train_sequence * (hidden * heads)
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

        # grafting
        # base = decoder_features[:, :, -1:]
        # outputs += base

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            outputs = outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            outputs = outputs + x_mean
        return outputs
