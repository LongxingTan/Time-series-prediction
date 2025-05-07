"""
`Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
<https://arxiv.org/abs/2012.07436>`_
"""

from typing import Any, Dict, Optional

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

from tfts.layers.attention_layer import Attention, ProbAttention
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding
from tfts.layers.mask_layer import CausalMask

from .base import BaseConfig, BaseModel


class InformerConfig(BaseConfig):
    model_type: str = "informer"

    def __init__(
        self,
        hidden_size=64,
        num_layers=1,
        num_decoder_layers=None,
        num_attention_heads=1,
        attention_probs_dropout_prob=0.0,
        ffn_intermediate_size=128,
        hidden_dropout_prob=0.0,
        prob_attention=False,
        distil_conv=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.prob_attention = prob_attention
        self.distil_conv = distil_conv


class Informer(BaseModel):
    """Informer model for time series"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        config: Optional[InformerConfig] = None,
    ):
        super(Informer, self).__init__()
        self.config = config or InformerConfig()
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = DataEmbedding(self.config.hidden_size)
        self.decoder_embedding = DataEmbedding(self.config.hidden_size)
        if not config.prob_attention:
            attn_layer = Attention(
                self.config.hidden_size, self.config.num_attention_heads, self.config.attention_probs_dropout_prob
            )
        else:
            attn_layer = ProbAttention(
                self.config.hidden_size, self.config.num_attention_heads, self.config.attention_probs_dropout_prob
            )
        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    attn_layer=attn_layer,
                    hidden_size=self.config.hidden_size,
                    hidden_dropout_prob=self.config.hidden_dropout_prob,
                    ffn_intermediate_size=self.config.ffn_intermediate_size,
                )
                for _ in range(self.config.num_layers)
            ],
            conv_layers=[DistilConv(filters=self.config.hidden_size) for _ in range(self.config.num_layers - 1)],
            norm_layer=LayerNormalization(),
        )

        if not config.prob_attention:
            attn_layer1 = Attention(
                self.config.hidden_size, self.config.num_attention_heads, self.config.attention_probs_dropout_prob
            )
        else:
            attn_layer1 = ProbAttention(
                self.config.hidden_size, self.config.num_attention_heads, self.config.attention_probs_dropout_prob
            )

        attn_layer2 = Attention(
            self.config.hidden_size, self.config.num_attention_heads, self.config.attention_probs_dropout_prob
        )
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    attn_layer1=attn_layer1,
                    attn_layer2=attn_layer2,
                    hidden_size=self.config.hidden_size,
                    hidden_dropout_prob=self.config.hidden_dropout_prob,
                    ffn_intermediate_size=self.config.ffn_intermediate_size,
                )
                for _ in range(self.config.num_decoder_layers)
            ]
        )
        self.projection = Dense(1)
        # self.projection = Dense(predict_sequence_length, activation=None)

    def __call__(
        self,
        inputs: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Informer call function"""
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=False)
        encoder_feature = self.encoder_embedding(encoder_feature)  # batch * seq * embedding_size
        memory = self.encoder(encoder_feature, mask=None)

        B, L, _ = tf.shape(decoder_feature)
        casual_mask = CausalMask(B * self.config.num_attention_heads, L).mask
        decoder_feature = self.decoder_embedding(decoder_feature)

        outputs = self.decoder(decoder_feature, memory=memory, x_mask=casual_mask)
        outputs = self.projection(outputs)

        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layers, conv_layers=None, norm_layer=None) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm_layer = norm_layer

    def call(self, x, mask=None):
        """Informer encoder call function"""
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.layers, self.conv_layers):
                x = attn_layer(x, mask)
                # x = conv_layer(x)
            x = self.layers[-1](x, mask)

        else:
            for attn_layer in self.layers:
                x = attn_layer(x, mask)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attn_layer, hidden_size, ffn_intermediate_size, hidden_dropout_prob) -> None:
        super().__init__()
        self.attn_layer = attn_layer
        self.hidden_size = hidden_size
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def build(self, input_shape):
        self.drop = Dropout(self.hidden_dropout_prob)
        self.norm1 = LayerNormalization()
        self.conv1 = Conv1D(filters=self.ffn_intermediate_size, kernel_size=1)
        self.conv2 = Conv1D(filters=self.hidden_size, kernel_size=1)
        self.norm2 = LayerNormalization()
        super(EncoderLayer, self).build(input_shape)

    def call(self, x, mask=None):
        """Informer encoder layer call function"""
        input = x
        x = self.attn_layer(x, x, x, mask)
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
            "hidden_size": self.hidden_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DistilConv(tf.keras.layers.Layer):
    def __init__(self, filters) -> None:
        super().__init__()
        self.filters = filters

    def build(self, input_shape):
        self.conv = Conv1D(filters=self.filters, kernel_size=3, padding="causal")
        self.norm = BatchNormalization()
        self.activation = Activation("elu")
        self.pool = MaxPool1D(pool_size=3, strides=2, padding="same")
        super().build(input_shape)

    def call(self, x):
        """Informer distil conv"""
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
        """Informer decoder call function"""
        for layer in self.layers:
            x = layer(x, memory, x_mask, memory_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attn_layer1, attn_layer2, hidden_size, ffn_intermediate_size, hidden_dropout_prob) -> None:
        super().__init__()
        self.attn1 = attn_layer1
        self.attn2 = attn_layer2
        self.hidden_size = hidden_size
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def build(self, input_shape):
        self.conv1 = Conv1D(filters=self.ffn_intermediate_size, kernel_size=1)
        self.conv2 = Conv1D(filters=self.hidden_size, kernel_size=1)
        self.drop = Dropout(self.hidden_dropout_prob)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()
        self.activation = Activation("relu")
        super(DecoderLayer, self).build(input_shape)

    def call(self, x, memory=None, x_mask=None, memory_mask=None):
        """Informer decoder layer call function"""
        x0 = x
        x = self.attn1(x, x, x, mask=x_mask)
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
            "hidden_size": self.hidden_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
