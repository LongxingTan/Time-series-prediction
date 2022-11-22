"""
`Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, TimeDistributed

from tfts.layers.attention_layer import FullAttention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding

params = {
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_token_embedding": False,
    "attention_hidden_sizes": 32 * 1,
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


class Transformer(object):
    """Transformer model"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        custom_model_params: Optional[Dict[str, Any]] = None,
        custom_model_head: Optional[Callable] = None,
    ):
        """Transformer for time series

        :param custom_model_params: custom model defined model hyper parameters
        :type custom_model_params: _dict_
        :param dynamic_decoding: _description_, defaults to True
        :type dynamic_decoding: bool, optional
        """
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = TokenEmbedding(params["attention_hidden_sizes"])

        self.encoder = Encoder(
            params["n_encoder_layers"],
            params["attention_hidden_sizes"],
            params["num_heads"],
            params["attention_dropout"],
            params["ffn_hidden_sizes"],
            params["ffn_filter_sizes"],
            params["ffn_dropout"],
        )
        # self.decoder = Decoder(
        #     predict_sequence_length,
        #     params['n_decoder_layers'],
        #     params['attention_hidden_sizes'],
        #     params['num_heads'],
        #     params['attention_dropout'],
        #     params['ffn_hidden_sizes'],
        #     params['ffn_filter_sizes'],
        #     params['ffn_dropout'])
        self.decoder = Decoder2(
            predict_sequence_length,
            embed_layer=TokenEmbedding(params["attention_hidden_sizes"]),
            att_layers=[
                DecoderLayer2(
                    params["n_decoder_layers"],
                    params["attention_hidden_sizes"],
                    params["num_heads"],
                    params["attention_dropout"],
                    params["ffn_hidden_sizes"],
                    params["ffn_filter_sizes"],
                    params["ffn_dropout"],
                )
                for _ in range(params["n_decoder_layers"])
            ],
            # norm_layer = LayerNormalization()
        )
        self.project = Dense(1, activation=None)

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
            decoder_feature = inputs["decoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs
            decoder_feature = tf.cast(
                tf.tile(
                    tf.reshape(tf.range(self.predict_sequence_length), (1, self.predict_sequence_length, 1)),
                    (tf.shape(encoder_feature)[0], 1, 1),
                ),
                tf.float32,
            )

        encoder_feature = self.encoder_embedding(encoder_feature)  # batch * seq * embedding_size
        memory = self.encoder(encoder_feature, src_mask=None)

        # decoder_outputs = self.decoder(decoder_features, init_input=x[:, -1:], encoder_memory=memory, teacher=teacher)
        decoder_outputs = self.decoder(decoder_feature, memory)
        decoder_outputs = self.project(decoder_outputs)

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, :]
            decoder_outputs = decoder_outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x, axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            decoder_outputs = decoder_outputs + x_mean
        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        n_encoder_layers,
        attention_hidden_sizes,
        num_heads,
        attention_dropout,
        ffn_hidden_sizes,
        ffn_filter_sizes,
        ffn_dropout,
    ):
        super(Encoder, self).__init__()
        self.n_encoder_layers = n_encoder_layers
        self.attention_hidden_sizes = attention_hidden_sizes
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_hidden_sizes = ffn_hidden_sizes
        self.ffn_filter_sizes = ffn_filter_sizes
        self.ffn_dropout = ffn_dropout
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.n_encoder_layers):
            attention_layer = SelfAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
            feed_forward_layer = FeedForwardNetwork(self.ffn_hidden_sizes, self.ffn_filter_sizes, self.ffn_dropout)
            ln_layer1 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=1e-6, dtype="float32")
            self.layers.append([attention_layer, ln_layer1, feed_forward_layer, ln_layer2])
        super(Encoder, self).build(input_shape)

    def call(self, encoder_inputs, src_mask=None):
        """_summary_

        Parameters
        ----------
        encoder_inputs : _type_
            _description_
        src_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        x = encoder_inputs
        for _, layer in enumerate(self.layers):
            attention_layer, ln_layer1, ffn_layer, ln_layer2 = layer
            enc = x
            enc = attention_layer(enc, src_mask)
            enc1 = ln_layer1(x + enc)  # residual connect
            enc1 = ffn_layer(enc1)
            x = ln_layer2(enc + enc1)
        return x

    def get_config(self):
        config = {
            "n_encoder_layers": self.n_encoder_layers,
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_filter_sizes": self.ffn_filter_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        predict_sequence_length,
        n_decoder_layers,
        attention_hidden_sizes,
        num_heads,
        attention_dropout,
        ffn_hidden_sizes,
        ffn_filter_sizes,
        ffn_dropout,
    ):
        super(Decoder, self).__init__()
        self.predict_sequence_length = predict_sequence_length
        self.decoder_embedding = DataEmbedding(embed_size=attention_hidden_sizes)
        self.decoder_layer = DecoderLayer(
            n_decoder_layers,
            attention_hidden_sizes,
            num_heads,
            attention_dropout,
            ffn_hidden_sizes,
            ffn_filter_sizes,
            ffn_dropout,
        )
        self.projection = Dense(units=1, name="final_projection")

    def call(
        self, decoder_features, init_input, encoder_memory, teacher=None, scheduler_sampling=0, training=None, **kwargs
    ):
        """_summary_

        Parameters
        ----------
        decoder_features : _type_
            _description_
        init_input : _type_
            _description_
        encoder_memory : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None
        scheduler_sampling : int, optional
            _description_, by default 0
        training : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        this_input = init_input

        for i in range(self.predict_sequence_length):
            if training:
                p = np.random.uniform(low=0, high=1, size=1)[0]
                if teacher is not None and p > scheduler_sampling:
                    input = teacher[:, : i + 1]
                else:
                    input = this_input[:, : i + 1]
            else:
                input = this_input[:, : i + 1]

            if decoder_features is not None:
                input = tf.concat([input, decoder_features[:, : i + 1]], axis=-1)

            embed_input = self.decoder_embedding(input)
            this_output = self.decoder_layer(embed_input, encoder_memory, tgt_mask=None)
            this_output = self.projection(this_output)
            this_input = tf.concat([this_input, this_output[:, -1:, :]], axis=1)
        return this_input[:, 1:]

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_decoder_layers,
        attention_hidden_sizes,
        num_heads,
        attention_dropout,
        ffn_hidden_sizes,
        ffn_filter_sizes,
        ffn_dropout,
        eps=1e-7,
    ):
        super(DecoderLayer, self).__init__()
        self.n_decoder_layers = n_decoder_layers
        self.attention_hidden_sizes = attention_hidden_sizes
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_hidden_sizes = ffn_hidden_sizes
        self.ffn_filter_sizes = ffn_filter_sizes
        self.ffn_dropout = ffn_dropout
        self.eps = eps
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.n_decoder_layers):
            self_attention_layer = SelfAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
            enc_dec_attention_layer = FullAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
            feed_forward_layer = FeedForwardNetwork(self.ffn_hidden_sizes, self.ffn_filter_sizes, self.ffn_dropout)
            ln_layer1 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=self.eps, dtype="float32")
            self.layers.append(
                [self_attention_layer, enc_dec_attention_layer, feed_forward_layer, ln_layer1, ln_layer2, ln_layer3]
            )
        super(DecoderLayer, self).build(input_shape)

    def call(self, decoder_inputs, encoder_memory, tgt_mask=None, cross_mask=None):
        """_summary_

        Parameters
        ----------
        decoder_inputs : _type_
            _description_
        encoder_memory : _type_
            _description_
        tgt_mask : _type_, optional
            _description_, by default None
        cross_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        x = decoder_inputs

        for _, layer in enumerate(self.layers):
            self_attention_layer, enc_dec_attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3 = layer
            dec = x
            dec = self_attention_layer(dec, mask=tgt_mask)
            dec1 = ln_layer1(x + dec)
            dec1 = enc_dec_attention_layer(dec1, encoder_memory, encoder_memory, mask=cross_mask)
            dec2 = ln_layer2(dec + dec1)
            dec2 = ffn_layer(dec2)
            x = ln_layer3(dec1 + dec2)
        return x

    def get_config(self):
        config = {
            "n_decoder_layers": self.n_decoder_layers,
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_filter_sizes": self.ffn_filter_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder2(tf.keras.layers.Layer):
    def __init__(self, predict_sequence_length, embed_layer, att_layers, norm_layer=None) -> None:
        super().__init__()
        self.predict_sequence_length = predict_sequence_length
        self.att_layers = att_layers
        self.norm = norm_layer
        self.decoder_embedding = embed_layer
        self.drop = Dropout(0.2)
        # self.dense1 = TimeDistributed(Dense(256))
        # self.drop1 = TimeDistributed(Dropout(0.2))
        self.dense2 = TimeDistributed(Dense(32))
        self.drop2 = TimeDistributed(Dropout(0.1))
        self.proj = TimeDistributed(Dense(1))

    def decode(self, x, cross, x_mask, cross_mask):
        x = self.decoder_embedding(x)
        for layer in self.att_layers:
            x = layer(x, cross, x_mask, cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def call(self, x, cross, x_mask=None, cross_mask=None, training=True):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        cross : _type_
            _description_
        x_mask : _type_, optional
            _description_, by default None
        cross_mask : _type_, optional
            _description_, by default None
        training : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        x = self.decode(x, cross, x_mask, cross_mask)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.proj(x)

        # if training:
        #     x = self.decode(x, cross, x_mask, cross_mask)
        # else:
        # for _ in range(self.predict_sequence_length):
        #     x1 = self.decode(x, cross, x_mask, cross_mask)
        #     print(x1.shape, x.shape)
        #     x = tf.concat([x, x1], axis=0)
        return x


class DecoderLayer2(tf.keras.layers.Layer):
    def __init__(
        self,
        n_decoder_layers,
        attention_hidden_sizes,
        num_heads,
        attention_dropout,
        ffn_hidden_sizes,
        ffn_filter_sizes,
        ffn_dropout,
        eps=1e-7,
    ):
        super(DecoderLayer2, self).__init__()
        self.n_decoder_layers = n_decoder_layers
        self.attention_hidden_sizes = attention_hidden_sizes
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_hidden_sizes = ffn_hidden_sizes
        self.ffn_filter_sizes = ffn_filter_sizes
        self.ffn_dropout = ffn_dropout
        self.eps = eps
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.n_decoder_layers):
            self_attention_layer = SelfAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
            enc_dec_attention_layer = FullAttention(self.attention_hidden_sizes, self.num_heads, self.attention_dropout)
            feed_forward_layer = FeedForwardNetwork(self.ffn_hidden_sizes, self.ffn_filter_sizes, self.ffn_dropout)
            ln_layer1 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=self.eps, dtype="float32")
            self.layers.append(
                [self_attention_layer, enc_dec_attention_layer, feed_forward_layer, ln_layer1, ln_layer2, ln_layer3]
            )
        super(DecoderLayer2, self).build(input_shape)

    def call(self, decoder_inputs, encoder_memory, tgt_mask=None, cross_mask=None):
        """_summary_

        Parameters
        ----------
        decoder_inputs : _type_
            _description_
        encoder_memory : _type_
            _description_
        tgt_mask : _type_, optional
            _description_, by default None
        cross_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        x = decoder_inputs

        for _, layer in enumerate(self.layers):
            self_attention_layer, enc_dec_attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3 = layer
            dec1 = x
            # dec = self_attention_layer(dec, mask=tgt_mask)
            # dec1 = ln_layer1(x + dec)
            dec1 = enc_dec_attention_layer(dec1, encoder_memory, encoder_memory, mask=cross_mask)
            dec2 = ln_layer2(x + dec1)
            dec2 = ffn_layer(dec2)
            # x = ln_layer3(dec1 + dec2)
            x = dec1 + dec2
        return x

    def get_config(self):
        config = {
            "n_decoder_layers": self.n_decoder_layers,
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_filter_sizes": self.ffn_filter_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(DecoderLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
