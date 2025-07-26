"""
`Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention

from tfts.layers.attention_layer import Attention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding
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
        positional_type: str = "positional encoding",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
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
            positional_type: The type of position embeddings (absolute or relative).
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
        self.positional_type: str = positional_type
        self.use_cache: bool = use_cache
        self.classifier_dropout: Optional[float] = classifier_dropout
        self.layer_norm_eps: float = layer_norm_eps
        self.pad_token_id: int = pad_token_id


class Transformer(BaseModel):
    """Transformer model"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[TransformerConfig] = None) -> None:
        """Transformer for time series"""
        super(Transformer, self).__init__()
        self.config = config or TransformerConfig()
        self.predict_sequence_length = predict_sequence_length
        self.encoder_embedding = DataEmbedding(self.config.hidden_size, positional_type=self.config.positional_type)

        self.encoder = Encoder(
            num_hidden_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            layer_norm_eps=self.config.layer_norm_eps,
        )

        self.decoder = Decoder(
            predict_sequence_length=predict_sequence_length,
            num_decoder_layers=self.config.num_decoder_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            layer_norm_eps=self.config.layer_norm_eps,
        )

    def __call__(
        self,
        inputs: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
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

        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=False)

        encoder_feature = self.encoder_embedding(encoder_feature)  # batch * seq * embedding_size
        memory = self.encoder(encoder_feature, mask=None)
        decoder_outputs = self.decoder(
            decoder_feature, init_input=x[:, -1:, 0:1], encoder_memory=memory, teacher=teacher
        )

        # Example for new CausalMask usage:
        # dummy = tf.zeros((B, L, 1))
        # mask_layer = CausalMask(num_attention_heads=1)
        # casual_mask = mask_layer(dummy)

        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float = 1e-9,
        **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.layers: List[List[tf.keras.layers.Layer]] = []

    def build(self, input_shape: Tuple[int]) -> None:
        for _ in range(self.num_hidden_layers):
            attention_layer = SelfAttention(
                self.hidden_size, self.num_attention_heads, self.attention_probs_dropout_prob
            )
            ffn_layer = FeedForwardNetwork(
                self.hidden_size,
                intermediate_size=self.ffn_intermediate_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
            )
            ln_layer1 = LayerNormalization(epsilon=self.layer_norm_eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.layer_norm_eps, dtype="float32")
            self.layers.append([attention_layer, ln_layer1, ffn_layer, ln_layer2])
        super(Encoder, self).build(input_shape)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None):
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
        x = inputs
        for _, layer in enumerate(self.layers):
            attention_layer, ln_layer1, ffn_layer, ln_layer2 = layer
            x = ln_layer1(x + attention_layer(x, mask=mask))
            x = ln_layer2(x + ffn_layer(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                "ffn_intermediate_size": self.ffn_intermediate_size,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


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
        layer_norm_eps: float = 1e-9,
        **kwargs
    ) -> None:
        super(Decoder, self).__init__(**kwargs)
        self.predict_sequence_length = predict_sequence_length
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        """Build the decoder layers."""
        super().build(input_shape)
        self.decoder_embedding = DataEmbedding(embed_size=self.hidden_size)
        self.decoder_layer = DecoderLayer(
            num_decoder_layers=self.num_decoder_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            ffn_intermediate_size=self.ffn_intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "predict_sequence_length": self.predict_sequence_length,
                "num_decoder_layers": self.num_decoder_layers,
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                "ffn_intermediate_size": self.ffn_intermediate_size,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_decoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float = 1e-9,
        **kwargs
    ) -> None:
        super(DecoderLayer, self).__init__(**kwargs)
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
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
            ln_layer1 = LayerNormalization(epsilon=self.layer_norm_eps, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=self.layer_norm_eps, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=self.layer_norm_eps, dtype="float32")
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

    def compute_output_shape(self, input_shape):
        return input_shape


class TransformerBlock(tf.keras.layers.Layer):
    """Basic Transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        embed_dim: int,
        feat_dim: int,
        num_heads: int,
        ffn_intermediate_size: int,
        rate: float = 0.1,
        layer_norm_eps: float = 1e-9,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.ffn_intermediate_size = ffn_intermediate_size
        self.rate = rate
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        """Build the Transformer block layers."""
        super().build(input_shape)
        self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = tf.keras.Sequential([Dense(self.ffn_intermediate_size, activation="gelu"), Dense(self.feat_dim)])
        self.layernorm1 = LayerNormalization(epsilon=self.layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=self.layer_norm_eps)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Forward pass through a Transformer block for time series."""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
