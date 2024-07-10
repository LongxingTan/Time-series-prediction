"""
`Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, TimeDistributed

from tfts.layers.attention_layer import FullAttention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding
from tfts.layers.mask_layer import CausalMask

from .base import BaseConfig, BaseModel

logger = logging.getLogger(__name__)


class TransformerConfig(BaseConfig):
    model_type = "transformer"

    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        num_decoder_layers=None,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        scheduled_sampling=1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super(TransformerConfig, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.scheduled_sampling = scheduled_sampling  # 0 means teacher forcing, 1 means use last prediction
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id


class Transformer(BaseModel):
    """Transformer model"""

    def __init__(self, predict_sequence_length: int = 1, config=TransformerConfig()) -> None:
        """Transformer for time series

        :param custom_model_config: custom model defined model hyper parameters
        :type custom_model_config: _dict_
        :param dynamic_decoding: _description_, defaults to True
        :type dynamic_decoding: bool, optional
        """
        super(Transformer, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = DataEmbedding(self.config.hidden_size)

        self.encoder = Encoder(
            num_hidden_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.decoder = Decoder2(
            embed_layer=DataEmbedding(config.hidden_size),
            att_layers=[
                DecoderLayer2(
                    num_decoder_layers=config.num_decoder_layers,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    intermediate_size=config.intermediate_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                )
                for _ in range(config.num_decoder_layers)
            ],
            # norm_layer = LayerNormalization()
        )
        self.project = Dense(1, activation=None)

    def __call__(self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None):
        """Time series transformer

        Parameters
        ----------
        inputs : tf.Tensor
            3D tensor for batch * seq_len * features
        teacher : tf.Tensor, optional
            _description_, by default None

        Returns
        -------
        tf.Tensor
            3D tensor for output, batch * output_seq * 1
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
        memory = self.encoder(encoder_feature, encoder_mask=None)

        # decoder_outputs = self.decoder(decoder_features, init_input=x[:, -1:], encoder_memory=memory, teacher=teacher)

        B, L, _ = tf.shape(decoder_feature)
        casual_mask = CausalMask(B * self.config.num_attention_heads, L).mask
        decoder_outputs = self.decoder(decoder_feature, memory, x_mask=casual_mask)
        decoder_outputs = self.project(decoder_outputs)

        return decoder_outputs

    def _shift_right(self, input_ids):
        return  # shifted_input_ids


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        intermediate_size: int,
        hidden_dropout_prob: float,
    ):
        super(Encoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layers: List[tf.keras.layers.Layer] = []

    def build(self, input_shape: Tuple[int]) -> None:
        for _ in range(self.num_hidden_layers):
            attention_layer = SelfAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            ffn_layer = FeedForwardNetwork(self.hidden_size, self.intermediate_size, self.hidden_dropout_prob)
            ln_layer1 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=1e-6, dtype="float32")
            self.layers.append([attention_layer, ln_layer1, ffn_layer, ln_layer2])
        super(Encoder, self).build(input_shape)

    def call(self, encoder_inputs: tf.Tensor, encoder_mask: Optional[tf.Tensor] = None):
        """Transformer encoder

        Parameters
        ----------
        inputs : tf.Tensor
            Transformer encoder inputs, with dimension of (batch, seq_len, features)
        mask : tf.Tensor, optional
            encoder mask to ignore it during attention, by default None

        Returns
        -------
        tf.Tensor
            Transformer encoder output
        """
        x = encoder_inputs
        for _, layer in enumerate(self.layers):
            attention_layer, ln_layer1, ffn_layer, ln_layer2 = layer
            enc = x
            enc = attention_layer(enc, encoder_mask)
            enc = ln_layer1(x + enc)  # residual connect
            enc1 = ffn_layer(enc)
            x = ln_layer2(enc + enc1)
        return x

    def get_config(self):
        config = {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        predict_sequence_length: int,
        n_decoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        intermediate_size: int,
        hidden_dropout_prob: float,
    ) -> None:
        super(Decoder, self).__init__()
        self.predict_sequence_length = predict_sequence_length
        self.decoder_embedding = DataEmbedding(embed_size=hidden_size)
        self.decoder_layer = DecoderLayer(
            n_decoder_layers,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            intermediate_size,
            hidden_dropout_prob,
        )
        self.projection = Dense(units=1, name="final_projection")

    def call(
        self, decoder_features, init_input, encoder_memory, teacher=None, scheduled_sampling=0, training=None, **kwargs
    ):
        """Transformer decoder

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
        scheduled_sampling : int, optional
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
                if teacher is not None and p > scheduled_sampling:
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

    def get_causal_attention_mask(self, inputs: tf.Tensor):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_decoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        intermediate_size: int,
        hidden_dropout_prob: float,
        eps: float = 1e-7,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.n_decoder_layers = n_decoder_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.eps = eps
        self.layers: List[tf.keras.layers.Layer] = []

    def build(self, input_shape):
        for _ in range(self.n_decoder_layers):
            self_attention_layer = SelfAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            attention_layer = FullAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            ffn_layer = FeedForwardNetwork(self.intermediate_size, self.hidden_size, self.hidden_dropout_prob)
            ln_layer1 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=self.eps, dtype="float32")
            self.layers.append([self_attention_layer, attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3])
        super(DecoderLayer, self).build(input_shape)

    def call(self, decoder_inputs, encoder_memory, tgt_mask=None, cross_mask=None, return_dict: Optional[bool] = None):
        """Decoder layer

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
            self_attention_layer, attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3 = layer
            dec = x
            dec = self_attention_layer(dec, mask=tgt_mask)
            dec = ln_layer1(x + dec)
            dec1 = attention_layer(dec, encoder_memory, encoder_memory, mask=cross_mask)
            dec1 = ln_layer2(dec + dec1)
            dec2 = ffn_layer(dec1)
            x = ln_layer3(dec1 + dec2)
        return x

    def get_config(self):
        config = {
            "n_decoder_layers": self.n_decoder_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder2(tf.keras.layers.Layer):
    def __init__(self, embed_layer, att_layers, norm_layer=None) -> None:
        super().__init__()
        self.att_layers = att_layers
        self.norm = norm_layer
        self.decoder_embedding = embed_layer
        self.drop = Dropout(0.2)
        # self.dense1 = TimeDistributed(Dense(256))
        # self.drop1 = TimeDistributed(Dropout(0.2))
        self.dense2 = TimeDistributed(Dense(32))
        self.drop2 = TimeDistributed(Dropout(0.1))
        self.proj = TimeDistributed(Dense(1))

    def call(self, x: tf.Tensor, memory, x_mask=None, memory_mask=None):
        """Transformer decoder2

        Parameters
        ----------
        x : _type_
            _description_
        memory : _type_
            _description_
        x_mask : _type_, optional
            _description_, by default None
        memory_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        tf.Tensor
            _description_
        """
        x = self.decoder_embedding(x)
        for layer in self.att_layers:
            x = layer(x, memory, x_mask, memory_mask)
        if self.norm is not None:
            x = self.norm(x)

        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.proj(x)
        return x


class DecoderLayer2(tf.keras.layers.Layer):
    def __init__(
        self,
        num_decoder_layers,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        intermediate_size,
        hidden_dropout_prob,
        eps=1e-7,
    ):
        super(DecoderLayer2, self).__init__()
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.eps = eps
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.num_decoder_layers):
            self_attention_layer = SelfAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            enc_dec_attention_layer = FullAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            feed_forward_layer = FeedForwardNetwork(self.hidden_size, self.intermediate_size, self.hidden_dropout_prob)
            ln_layer1 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=self.eps, dtype="float32")
            self.layers.append(
                [self_attention_layer, enc_dec_attention_layer, feed_forward_layer, ln_layer1, ln_layer2, ln_layer3]
            )
        super(DecoderLayer2, self).build(input_shape)

    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_memory: tf.Tensor,
        decoder_mask: Optional[tf.Tensor] = None,
        memory_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Decoder layer2

        Parameters
        ----------
        decoder_inputs : tf.Tensor
            _description_
        encoder_memory : _type_
            _description_
        decoder_mask : _type_, optional
            _description_, by default None
        memory_mask : _type_, optional
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
            dec = self_attention_layer(dec, mask=decoder_mask)
            dec1 = ln_layer1(x + dec)
            dec1 = enc_dec_attention_layer(dec1, encoder_memory, encoder_memory, mask=memory_mask)
            dec2 = ln_layer2(x + dec1)
            dec2 = ffn_layer(dec2)
            x = ln_layer3(dec1 + dec2)  # note that don't repeat ln
            # x = dec1 + dec2
        return x

    def get_config(self):
        config = {
            "n_decoder_layers": self.n_decoder_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(DecoderLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
