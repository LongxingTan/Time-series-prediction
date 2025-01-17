"""
`Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_
"""

import logging
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, TimeDistributed

from tfts.layers.attention_layer import Attention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding
from tfts.layers.mask_layer import CausalMask

from .base import BaseConfig, BaseModel

logger = logging.getLogger(__name__)


class TransformerConfig(BaseConfig):
    model_type: str = "transformer"

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_decoder_layers: int = 4,
        num_attention_heads: int = 4,
        num_kv_heads: int = 4,
        ffn_intermediate_size: int = 256,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        scheduled_sampling: float = 1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "positional encoding",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        **kwargs: Dict[str, object]
    ) -> None:
        """
        Initializes the configuration for the Transformer model with the specified parameters.

        Args:
            hidden_size: The size of the hidden layers.
            num_layers: The number of encoder layers.
            num_decoder_layers: The number of decoder layers.
            num_attention_heads: The number of attention heads.
            num_kv_heads: The number of key-value heads.
            ffn_intermediate_size: The size of the intermediate feed-forward layers.
            hidden_act: The activation function for hidden layers.
            hidden_dropout_prob: The dropout probability for hidden layers.
            attention_probs_dropout_prob: The dropout probability for attention probabilities.
            scheduled_sampling: Controls the use of teacher forcing vs. last prediction.
            max_position_embeddings: The maximum length of input sequences.
            initializer_range: The standard deviation for weight initialization.
            layer_norm_eps: The epsilon for layer normalization.
            pad_token_id: The ID for the padding token.
            position_embedding_type: The type of position embeddings (absolute or relative).
            use_cache: Whether to use cache during inference.
            classifier_dropout: Dropout rate for classifier layers.
            **kwargs: Additional parameters for further customization passed to the parent class.
        """
        super(TransformerConfig, self).__init__()

        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_decoder_layers: int = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_attention_heads: int = num_attention_heads
        self.num_kv_heads: int = num_kv_heads
        self.ffn_intermediate_size: int = ffn_intermediate_size
        self.hidden_act: str = hidden_act
        self.hidden_dropout_prob: float = hidden_dropout_prob
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob
        self.scheduled_sampling: float = scheduled_sampling
        self.max_position_embeddings: int = max_position_embeddings
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.position_embedding_type: str = position_embedding_type
        self.use_cache: bool = use_cache
        self.classifier_dropout: Optional[float] = classifier_dropout
        self.pad_token_id: int = pad_token_id


class Transformer(BaseModel):
    """Transformer model"""

    def __init__(self, predict_sequence_length: int = 1, config=None) -> None:
        """Transformer for time series"""
        super(Transformer, self).__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = DataEmbedding(self.config.hidden_size)

        self.encoder = Encoder(
            num_hidden_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            ffn_intermediate_size=config.ffn_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.decoder = Decoder(
            predict_sequence_length=predict_sequence_length,
            num_decoder_layers=config.num_decoder_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            ffn_intermediate_size=config.ffn_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

    def __call__(self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None, return_dict: Optional[bool] = None):
        """Time series transformer

        Parameters
        ----------
        inputs : tf.Tensor
            3D tensor for batch * seq_len * features
        teacher : tf.Tensor, optional
            the teacher for decoding, by default None
        return_dict: bool
            if return output a dict

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

        decoder_outputs = self.decoder(
            decoder_feature, init_input=x[:, -1:, 0:1], encoder_memory=memory, teacher=teacher
        )

        # B, L, _ = tf.shape(decoder_feature)
        # casual_mask = CausalMask(B, L).mask
        # decoder_outputs = self.decoder(decoder_feature, memory, x_mask=casual_mask)
        # decoder_outputs = self.project(decoder_outputs)

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
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
    ):
        super(Encoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layers: List[tf.keras.layers.Layer] = []

    def build(self, input_shape: Tuple[int]) -> None:
        for _ in range(self.num_hidden_layers):
            attention_layer = SelfAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            ffn_layer = FeedForwardNetwork(self.hidden_size, self.ffn_intermediate_size, self.hidden_dropout_prob)
            ln_layer1 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=1e-6, dtype="float32")
            self.layers.append([attention_layer, ln_layer1, ffn_layer, ln_layer2])
        super(Encoder, self).build(input_shape)

    def call(self, encoder_inputs: tf.Tensor, encoder_mask: Optional[tf.Tensor] = None):
        """Transformer encoder

        Parameters
        ----------
        encoder_inputs : tf.Tensor
            Transformer encoder inputs, with dimension of (batch, seq_len, features)
        encoder_mask : tf.Tensor, optional
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
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):
    """Transformer Decoder that supports both one-time and distributed decoding strategies."""

    def __init__(
        self,
        predict_sequence_length: int,
        num_decoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
    ) -> None:
        super(Decoder, self).__init__()
        self.predict_sequence_length = predict_sequence_length
        self.decoder_embedding = DataEmbedding(embed_size=hidden_size)
        self.decoder_layer = DecoderLayer(
            num_decoder_layers,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            ffn_intermediate_size,
            hidden_dropout_prob,
        )
        self.projection = Dense(units=1, name="final_projection")

    def call(
        self,
        decoder_features: tf.Tensor,
        init_input: tf.Tensor,
        encoder_memory: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        scheduled_sampling: float = 0.0,
        training: bool = False,
        **kwargs
    ):
        """Transformer decoder"""
        input_x = init_input
        if teacher is not None:
            teacher = tf.squeeze(teacher, 2)
            teachers = tf.split(teacher, self.predict_sequence_length, axis=1)
        else:
            teachers = None

        for i in range(self.predict_sequence_length):
            input_tensor = self._get_input_for_step(
                input_x, decoder_features, i, teachers, scheduled_sampling, training
            )
            embed_input = self.decoder_embedding(input_tensor)
            decoder_output = self.decoder_layer(embed_input, encoder_memory)
            projected_output = self.projection(decoder_output)
            input_x = tf.concat([input_x, projected_output[:, -1:, :]], axis=1)

        return input_x[:, 1:]  # Exclude the first token

    def _get_input_for_step(
        self,
        input_x: tf.Tensor,
        decoder_features: tf.Tensor,
        step: int,
        teachers: Optional[tf.Tensor],
        scheduled_sampling: float,
        training: bool,
    ) -> tf.Tensor:
        """Determine the input for each decoding step, considering teacher forcing and scheduled sampling."""

        if training:
            p = np.random.uniform(low=0, high=1)
            if teachers is not None and p > scheduled_sampling:
                this_input = teachers[step]
            else:
                this_input = input_x[:, : step + 1]
        else:
            this_input = input_x[:, : step + 1]

        if decoder_features is not None:
            input_tensor = tf.concat([this_input, decoder_features[:, : step + 1, :]], axis=-1)
        else:
            input_tensor = this_input

        return input_tensor

    def get_causal_attention_mask(self, sequence_length: int) -> tf.Tensor:
        """Generate a causal attention mask to ensure each token only attends to previous tokens."""
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        return tf.reshape(mask, (1, sequence_length, sequence_length))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_decoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
        eps: float = 1e-7,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.eps = eps
        self.layers: List[List[tf.keras.layers.Layer]] = []

    def build(self, input_shape):
        for _ in range(self.num_decoder_layers):
            self_attention_layer = SelfAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            cross_attention_layer = Attention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            ffn_layer = FeedForwardNetwork(self.ffn_intermediate_size, self.hidden_size, self.hidden_dropout_prob)
            ln_layer1 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.eps, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=self.eps, dtype="float32")
            self.layers.append(
                [self_attention_layer, cross_attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3]
            )
        super(DecoderLayer, self).build(input_shape)

    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_memory: tf.Tensor,
        tgt_mask: Optional[tf.Tensor] = None,
        cross_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Forward pass through the decoder layer."""
        x = decoder_inputs

        for self_attention_layer, attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3 in self.layers:
            x = ln_layer1(x + self_attention_layer(x, mask=tgt_mask))
            x = ln_layer2(x + attention_layer(x, encoder_memory, encoder_memory, mask=cross_mask))
            x = ln_layer3(x + ffn_layer(x))
        return x

    def get_config(self):
        config = {
            "n_decoder_layers": self.num_decoder_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerBlock(tf.keras.layers.Layer):
    """Basic Transformer block with attention and feed-forward layers."""

    def __init__(
        self, embed_dim: int, feat_dim: int, num_heads: int, ffn_intermediate_size: int, rate: float = 0.1
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ffn_intermediate_size, activation="gelu"), Dense(feat_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Forward pass through a Transformer block for time series."""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
