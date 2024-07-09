"""
`Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
<https://arxiv.org/abs/2012.07436>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

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

from tfts.layers.attention_layer import FullAttention, ProbAttention
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding
from tfts.layers.mask_layer import CausalMask

from .base import BaseConfig, BaseModel


class InformerConfig(BaseConfig):
    model_type = "informer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
    ):
        super().__init__()


config: Dict[str, Any] = {
    "num_hidden_layers": 1,
    "n_decoder_layers": 1,
    "hidden_size": 32 * 1,
    "num_attention_heads": 1,
    "attention_dropout": 0.0,
    "intermediate_size": 32 * 1,
    "ffn_dropout": 0.0,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
    "prob_attention": False,
    "distil_conv": False,
}


class Informer(BaseModel):
    """Informer model for time series"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        config=InformerConfig,
    ):
        super(Informer, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = DataEmbedding(config["hidden_size"])
        self.decoder_embedding = DataEmbedding(config["hidden_size"])
        if not config["prob_attention"]:
            attn_layer = FullAttention(
                config["hidden_size"], config["num_attention_heads"], config["attention_dropout"]
            )
        else:
            attn_layer = ProbAttention(
                config["hidden_size"], config["num_attention_heads"], config["attention_dropout"]
            )
        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    attn_layer=attn_layer,
                    hidden_size=config["hidden_size"],
                    ffn_dropout=config["ffn_dropout"],
                    intermediate_size=config["intermediate_size"],
                )
                for _ in range(config["num_hidden_layers"])
            ],
            conv_layers=[DistilConv(filters=config["hidden_size"]) for _ in range(config["num_hidden_layers"] - 1)],
            norm_layer=LayerNormalization(),
        )

        if not config["prob_attention"]:
            attn_layer1 = FullAttention(
                config["hidden_size"], config["num_attention_heads"], config["attention_dropout"]
            )
        else:
            attn_layer1 = ProbAttention(
                config["hidden_size"], config["num_attention_heads"], config["attention_dropout"]
            )

        attn_layer2 = FullAttention(config["hidden_size"], config["num_attention_heads"], config["attention_dropout"])
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    attn_layer1=attn_layer1,
                    attn_layer2=attn_layer2,
                    hidden_size=config["hidden_size"],
                    ffn_dropout=config["ffn_dropout"],
                    intermediate_size=config["intermediate_size"],
                )
                for _ in range(config["n_decoder_layers"])
            ]
        )
        self.projection = Dense(1)
        # self.projection = Dense(predict_sequence_length, activation=None)

    def __call__(self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None, return_dict: Optional[bool] = None):
        """Informer call function"""
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
                tf.reshape(tf.range(self.predict_sequence_length), (-1, self.predict_sequence_length, 1)), tf.float32
            )

        encoder_feature = self.encoder_embedding(encoder_feature)  # batch * seq * embedding_size
        memory = self.encoder(encoder_feature, mask=None)

        B, L, _ = tf.shape(decoder_feature)
        casual_mask = CausalMask(B * self.config["num_attention_heads"], L).mask
        decoder_feature = self.decoder_embedding(decoder_feature)

        outputs = self.decoder(decoder_feature, memory=memory, x_mask=casual_mask)
        outputs = self.projection(outputs)

        if self.config["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            outputs = outputs + x_mean
        if self.config["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            outputs = outputs + x_mean
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
    def __init__(self, attn_layer, hidden_size, intermediate_size, ffn_dropout) -> None:
        super().__init__()
        self.attn_layer = attn_layer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ffn_dropout = ffn_dropout

    def build(self, input_shape):
        self.drop = Dropout(self.ffn_dropout)
        self.norm1 = LayerNormalization()
        self.conv1 = Conv1D(filters=self.intermediate_size, kernel_size=1)
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
            "intermediate_size": self.intermediate_size,
            "ffn_dropout": self.ffn_dropout,
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
    def __init__(self, attn_layer1, attn_layer2, hidden_size, intermediate_size, ffn_dropout) -> None:
        super().__init__()
        self.attn1 = attn_layer1
        self.attn2 = attn_layer2
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ffn_dropout = ffn_dropout

    def build(self, input_shape):
        self.conv1 = Conv1D(filters=self.intermediate_size, kernel_size=1)
        self.conv2 = Conv1D(filters=self.hidden_size, kernel_size=1)
        self.drop = Dropout(self.ffn_dropout)
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
            "intermediate_size": self.intermediate_size,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
