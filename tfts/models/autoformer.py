"""
`Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
<https://arxiv.org/abs/2106.13008>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization

from tfts.layers.attention_layer import FullAttention, SelfAttention
from tfts.layers.autoformer_layer import AutoCorrelation, SeriesDecomp
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding

params = {
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "kernel_size": 24,
    "attention_hidden_sizes": 32,
    "num_heads": 1,
    "attention_dropout": 0.0,
    "ffn_hidden_sizes": 32 * 1,
    "ffn_filter_sizes": 32 * 1,
    "ffn_dropout": 0.0,
    "layer_postprocess_dropout": 0.0,
    "scheduler_sampling": 1,  # 0 means teacher forcing, 1 means use last prediction
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class AutoFormer(object):
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

        # self.encoder_embedding = TokenEmbedding(params['attention_hidden_sizes'])
        self.series_decomp = SeriesDecomp(params["kernel_size"])
        self.encoder = [
            EncoderLayer(
                params["kernel_size"],
                params["attention_hidden_sizes"],
                params["num_heads"],
                params["attention_dropout"],
            )
            for _ in range(params["n_encoder_layers"])
        ]

        self.decoder = [
            DecoderLayer(
                params["kernel_size"],
                params["attention_hidden_sizes"],
                params["num_heads"],
                params["attention_dropout"],
            )
            for _ in range(params["n_decoder_layers"])
        ]

        self.project = Conv1D(1, kernel_size=3, strides=1, padding="same", use_bias=False)

        self.project1 = Dense(predict_sequence_length, activation=None)
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation="relu")
        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation="relu")

    def __call__(self, inputs, teacher=None, **kwargs):
        """autoformer call

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

        batch_size, _, n_feature = tf.shape(encoder_feature)
        # # de-comp
        # seasonal_init, trend_init = self.series_decomp(encoder_feature)
        # # decoder input
        # seasonal_init = tf.concat(
        #     [seasonal_init, tf.zeros([batch_size, self.predict_sequence_length, n_feature])], axis=1
        # )
        # trend_init = tf.concat(
        #     [
        #         trend_init,
        #         tf.repeat(
        #             tf.reduce_mean(encoder_feature, axis=1)[:, tf.newaxis, :],
        #             repeats=self.predict_sequence_length,
        #             axis=1,
        #         ),
        #     ],
        #     axis=1,
        # )

        # encoder
        for layer in self.encoder:
            x = layer(x)

        encoder_output = x[:, -1]
        encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.bn2(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)
        return outputs

        # encoder_output = x
        # trend_part= trend_init[..., 0:1]

        # for layer in self.decoder:
        #     seasonal_part, trend_part = layer(seasonal_init, encoder_output, trend_part)

        # trend_part = trend_part[:, -self.predict_sequence_length:, :]
        # seasonal_part = seasonal_part[:, -self.predict_sequence_length:, :]
        # seasonal_part = self.project(seasonal_part)
        # return trend_part + seasonal_part


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, d_model, num_heads, dropout_rate=0.1) -> None:
        super().__init__()
        self.series_decomp1 = SeriesDecomp(kernel_size)
        self.series_decomp2 = SeriesDecomp(kernel_size)
        self.autocorrelation = AutoCorrelation(d_model, num_heads)
        self.drop = Dropout(dropout_rate)

    def build(self, input_shape):
        self.dense = Dense(input_shape[-1])

    def call(self, x):
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
        x, _ = self.series_decomp1(self.autocorrelation(x, x, x) + x)
        x, _ = self.series_decomp2(self.drop(self.dense(x)) + x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, d_model, num_heads, drop_rate=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.drop_rate = drop_rate
        self.series_decomp1 = SeriesDecomp(kernel_size)
        self.series_decomp2 = SeriesDecomp(kernel_size)
        self.series_decomp3 = SeriesDecomp(kernel_size)
        self.autocorrelation1 = AutoCorrelation(d_model, num_heads)
        self.autocorrelation2 = AutoCorrelation(d_model, num_heads)

    def build(self, input_shape):
        self.conv1 = Conv1D(self.d_model, kernel_size=3, strides=1, padding="same")
        self.project = Conv1D(1, kernel_size=3, strides=1, padding="same")
        self.drop = Dropout(self.drop_rate)
        self.dense1 = Dense(input_shape[-1])
        self.conv2 = Conv1D(input_shape[-1], kernel_size=3, strides=1, padding="same")
        self.activation = tf.keras.activations.gelu

    def call(self, x, cross, init_trend):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        cross : _type_
            _description_
        init_trend : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x, trend1 = self.series_decomp1(self.drop(self.autocorrelation1(x, x, x)) + x)
        x, trend2 = self.series_decomp2(self.drop(self.autocorrelation2(x, cross, cross)) + x)
        x = self.conv2(self.drop(self.activation(self.conv1(x))))
        x, trend3 = self.series_decomp3(self.drop(self.dense1(x)) + x)

        trend = trend1 + trend2 + trend3
        trend = self.drop(self.project(trend))
        return x, init_trend + trend
