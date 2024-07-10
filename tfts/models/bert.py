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

from .base import BaseConfig, BaseModel


class BertConfig(BaseConfig):
    model_type = "bert"

    def __init__(
        self,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):

        super(BertConfig, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id


class Bert(BaseModel):
    def __init__(
        self,
        predict_sequence_length: int = 1,
        config=BertConfig(),
    ) -> None:
        super(Bert, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        self.encoder_embedding = TokenEmbedding(config.hidden_size)
        # self.spatial_drop = SpatialDropout1D(0.1)
        # self.tcn = ConvTemporal(kernel_size=2, filters=32, dilation_rate=6)
        self.encoder = Encoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.project1 = Dense(predict_sequence_length, activation=None)

        # self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation="relu")

        # self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation="relu")

        # self.forecasting = Forecasting(predict_sequence_length, self.config)
        # self.pool1 = AveragePooling1D(pool_size=6)
        # self.rnn1 = GRU(units=1, activation='tanh', return_state=False, return_sequences=True, dropout=0)
        # self.rnn2 = GRU(units=32, activation='tanh', return_state=False, return_sequences=True, dropout=0)
        # self.project2 = Dense(48, activation=None)
        # self.project3 = Dense(1, activation=None)
        #
        # self.dense_se = Dense(16, activation='relu')
        # self.dense_se2 = Dense(1, activation='sigmoid')

    def __call__(
        self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None, return_dict: Optional[bool] = None
    ) -> tf.Tensor:
        """Bert model call

        Parameters
        ----------
        inputs : tf.Tensor
            BERT model input
        teacher : tf.Tensor, optional
            teacher forcing for autoregression, by default None

        Returns
        -------
        tf.Tensor
            BERT model output tensor as prediction output
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

        # batch * train_sequence * (hidden * heads)
        memory = self.encoder(encoder_feature, encoder_mask=None)
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

        return outputs
