# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com


import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    LayerNormalization,
    MaxPool1D,
)

from tfts.layers.attention_layer import FullAttention, SelfAttention
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding

params = {
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "attention_hidden_sizes": 32 * 1,
    "num_heads": 1,
    "attention_dropout": 0.0,
    "ffn_hidden_sizes": 32 * 1,
    "ffn_filter_sizes": 32 * 1,
    "ffn_dropout": 0.0,
    "skip_connect": False,
}


class Informer(object):
    def __init__(self, predict_sequence_length=3, custom_model_params=None):
        """Transformer for time series
        :param custom_model_params: _description_
        :type custom_model_params: _type_
        :param dynamic_decoding: _description_, defaults to True
        :type dynamic_decoding: bool, optional
        """
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = TokenEmbedding(params["attention_hidden_sizes"])
        self.decoder_embedding = TokenEmbedding(params["attention_hidden_sizes"])
        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    attention_hidden_sizes=params["attention_hidden_sizes"],
                    num_heads=params["num_heads"],
                    attention_dropout=params["attention_dropout"],
                    ffn_dropout=params["ffn_dropout"],
                    ffn_hidden_sizes=params["ffn_hidden_sizes"],
                )
                for _ in range(params["n_encoder_layers"])
            ],
            # conv_layers = [
            #     CustomConv(
            #         filters=params['attention_hidden_sizes']
            #     ) for _ in range(params['n_encoder_layers'] - 1)
            # ],
            norm_layer=LayerNormalization(),
        )
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    attention_hidden_sizes=params["attention_hidden_sizes"],
                    num_heads=params["num_heads"],
                    attention_dropout=params["attention_dropout"],
                    ffn_dropout=params["ffn_dropout"],
                    ffn_hidden_sizes=params["ffn_hidden_sizes"],
                )
                for _ in range(params["n_decoder_layers"])
            ]
        )
        self.projection = Dense(1)
        # self.projection = Dense(predict_sequence_length, activation=None)

    def __call__(self, inputs, teacher=None):
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, decoder_features = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:
            encoder_features = x = inputs
            decoder_features = None

        encoder_features = self.encoder_embedding(encoder_features)  # batch * seq * embedding_size
        memory = self.encoder(encoder_features, mask=None)

        decoder_features = self.decoder_embedding(decoder_features)
        decoder_outputs = self.decoder(decoder_features, memory=memory)
        decoder_outputs = self.projection(decoder_outputs)
        # decoder_outputs = decoder_outputs[:, -self.predict_sequence_length:, :]

        if self.params["skip_connect"]:
            x_mean = tf.tile(tf.reduce_mean(x, axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            decoder_outputs = decoder_outputs + x_mean
        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layers, conv_layers=None, norm_layer=None) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm_layer = norm_layer

    def call(self, x, mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.layers, self.conv_layers):
                x = attn_layer(x, mask)
                x = conv_layer(x)
            x = self.layers[-1](x, mask)

        else:
            for attn_layer in self.layers:
                x = attn_layer(x, mask)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_hidden_sizes, num_heads, attention_dropout, ffn_dropout, ffn_hidden_sizes) -> None:
        super().__init__()
        self.attention_hidden_sizes = attention_hidden_sizes
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.ffn_hidden_sizes = ffn_hidden_sizes

    def build(self, input_shape):
        self.attn_layer = SelfAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
        self.drop = Dropout(self.ffn_dropout)
        self.norm1 = LayerNormalization()
        self.conv1 = Conv1D(filters=self.ffn_hidden_sizes, kernel_size=1)
        self.conv2 = Conv1D(filters=self.attention_hidden_sizes, kernel_size=1)
        self.norm2 = LayerNormalization()
        super(EncoderLayer, self).build(input_shape)

    def call(self, x, mask=None):
        input = x
        x = self.attn_layer(x, mask)
        x = self.drop(x)
        x = x + input

        y = x = self.norm1(x)
        y = self.conv1(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.drop(y)
        y = x + y
        y = self.norm2(y)
        return y

    def get_config(self):
        config = {
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CustomConv(tf.keras.layers.Layer):
    def __init__(self, filters) -> None:
        super().__init__()
        self.filters = filters

    def build(self, input_shape):
        self.conv = Conv1D(filters=self.filters, kernel_size=3, padding="same")
        self.norm = BatchNormalization()
        self.activation = Activation("elu")
        self.pool = MaxPool1D(pool_size=3, strides=2, padding="same")
        super(CustomConv, self).build(input_shape)

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = layers
        self.norm = norm_layer

    def call(self, x, memory=None, x_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, x_mask, memory_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_hidden_sizes, num_heads, attention_dropout, ffn_dropout, ffn_hidden_sizes) -> None:
        super().__init__()
        self.attention_hidden_sizes = attention_hidden_sizes
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.ffn_hidden_sizes = ffn_hidden_sizes

    def build(self, input_shape):
        self.attn1 = SelfAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
        self.attn2 = FullAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
        self.conv1 = Conv1D(filters=self.ffn_hidden_sizes, kernel_size=1)
        self.conv2 = Conv1D(filters=self.attention_hidden_sizes, kernel_size=1)
        self.drop = Dropout(self.ffn_dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()
        self.activation = Activation("relu")
        super(DecoderLayer, self).build(input_shape)

    def call(self, x, memory=None, x_mask=None, memory_mask=None):
        x0 = x
        x = self.attn1(x, x_mask)
        x = self.drop(x)
        x = x + x0
        x = self.norm1(x)

        x1 = x
        x = self.attn2(x, memory, memory, mask=memory_mask)
        x = self.drop(x)
        x = x + x1
        x = self.norm2(x)

        x2 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x + x2
        return self.norm3(x)

    def get_config(self):
        config = {
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
