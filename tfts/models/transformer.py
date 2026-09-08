"""
`Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_
"""

from dataclasses import replace
import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention

from tfts.contracts import ForecastOutput
from tfts.generation.decoders import _DecodeSession as DecodeSession, _StepOutput as StepOutput
from tfts.layers.attention_layer import SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding

from ._autoregressive import AUTOREGRESSIVE_CAPABILITIES, AutoregressiveModel, decoder_features, encoder_features
from .base import CommonConfig
from .registry import register_model

logger = logging.getLogger(__name__)


class TransformerConfig(CommonConfig):
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
        **kwargs: Any,
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

        self.decoder_format_version = 2

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

    @staticmethod
    def validate_checkpoint_config(values):
        if values.get("decoder_format_version") != 2:
            raise ValueError(
                "Incompatible Transformer decoder checkpoint: causal decoding requires format 2 "
                "(TFTS >= 0.1.0). Legacy decoder weights require migration or retraining."
            )


@register_model(
    "transformer",
    config=TransformerConfig,
    paper="https://arxiv.org/abs/1706.03762",
    tags=("attention", "encoder-decoder"),
    tier="core",
    capabilities=replace(AUTOREGRESSIVE_CAPABILITIES, supports_parallel_teacher_forcing=True),
)
class Transformer(AutoregressiveModel):
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
            target_dim=self.config.target_dim,
            use_cache=self.config.use_cache,
            predict_sequence_length=predict_sequence_length,
            num_decoder_layers=self.config.num_decoder_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            layer_norm_eps=self.config.layer_norm_eps,
        )

    def initialize_decode(self, batch, *, horizon, training=False):
        features = decoder_features(batch, horizon)
        memory = self.encoder(self.encoder_embedding(encoder_features(batch)), training=training)
        return DecodeSession(
            (features, memory),
            self.decoder.initialize_state(features, horizon),
            self.decoder_seed(batch),
        )

    def decode_step(self, previous, state, context, *, offset, training=False):
        features, memory = context
        return self.decoder.step(
            previous, state, features[:, offset : offset + 1, :], memory, offset=offset, training=training
        )

    def decode_teacher_forced(self, batch, *, training=False):
        horizon = tf.shape(batch.future_values)[1]
        features = decoder_features(batch, horizon)
        previous = tf.concat([self.decoder_seed(batch), batch.future_values[:, :-1, :]], axis=1)
        memory = self.encoder(self.encoder_embedding(encoder_features(batch)), training=training)
        return ForecastOutput(predictions=self.decoder.sequence(previous, features, memory, training=training))


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
        **kwargs,
    ):
        super(Encoder, self).__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.encoder_layers: List[tf.keras.layers.Layer] = []

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
            self.encoder_layers.extend([attention_layer, ln_layer1, ffn_layer, ln_layer2])
        super(Encoder, self).build(input_shape)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, training=None):
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
        for index in range(0, len(self.encoder_layers), 4):
            attention_layer, ln_layer1, ffn_layer, ln_layer2 = self.encoder_layers[index : index + 4]
            x = ln_layer1(x + attention_layer(x, mask=mask, training=training))
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


class CausalAttention(tf.keras.layers.Layer):
    """Shared attention weights for full-sequence and cached decoding."""

    def __init__(self, hidden_size, heads, dropout):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size, self.heads = hidden_size, heads
        self.q, self.k, self.v = [Dense(hidden_size, use_bias=False) for _ in range(3)]
        self.dropout = Dropout(dropout)

    def attend(self, query, key, value, mask, training):
        def split(x):
            shape = tf.shape(x)
            return tf.transpose(
                tf.reshape(x, [shape[0], shape[1], self.heads, self.hidden_size // self.heads]), [0, 2, 1, 3]
            )

        q, k, v = [split(x) for x in (query, key, value)]
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.hidden_size // self.heads, q.dtype))
        if mask is not None:
            scores = tf.where(mask[None, None, ...], scores, tf.cast(-1e9, scores.dtype))
        weights = self.dropout(tf.nn.softmax(scores), training=training)
        values = tf.transpose(tf.matmul(weights, v), [0, 2, 1, 3])
        return tf.reshape(values, [tf.shape(values)[0], tf.shape(values)[1], self.hidden_size])

    def call(self, x, memory=None, cache=None, offset=None, training=None):
        source = x if memory is None else memory
        q, k, v = self.q(x), self.k(source), self.v(source)
        mask = None
        if cache is not None:
            slot = tf.one_hot(offset, tf.shape(cache[0])[1], dtype=k.dtype)[None, :, None]
            k, v = cache[0] * (1 - slot) + k * slot, cache[1] * (1 - slot) + v * slot
            mask = tf.range(tf.shape(k)[1])[None, :] <= offset
        elif memory is None:
            positions = tf.range(tf.shape(x)[1])
            mask = positions[:, None] >= positions[None, :]
        return self.attend(q, k, v, mask, training), (k, v)


class DecoderLayer(tf.keras.layers.Layer):
    """One causal decoder block; state contains only projected self-attention keys/values."""

    def __init__(self, hidden_size, heads, attention_dropout, intermediate_size, dropout, epsilon):
        super().__init__()
        self.self_attention = CausalAttention(hidden_size, heads, attention_dropout)
        self.cross_attention = CausalAttention(hidden_size, heads, attention_dropout)
        self.ffn = tf.keras.Sequential(
            [Dense(intermediate_size, activation="gelu"), Dense(hidden_size), Dropout(dropout)]
        )
        self.norms = [LayerNormalization(epsilon=epsilon) for _ in range(3)]

    def call(self, x, memory, cache=None, offset=None, training=None):
        attention, next_cache = self.self_attention(x, cache=cache, offset=offset, training=training)
        x = self.norms[0](x + attention)
        attention, _ = self.cross_attention(x, memory=memory, training=training)
        x = self.norms[1](x + attention)
        return self.norms[2](x + self.ffn(x, training=training)), next_cache


class Decoder(tf.keras.layers.Layer):
    """Causal sequence and incremental execution over exactly the same weights."""

    def __init__(
        self,
        predict_sequence_length,
        num_decoder_layers,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        ffn_intermediate_size,
        hidden_dropout_prob,
        layer_norm_eps=1e-9,
        target_dim=1,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.predict_sequence_length = predict_sequence_length
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.target_dim = target_dim
        self.use_cache = use_cache
        self.embedding = Dense(hidden_size, use_bias=False)
        self.projection = Dense(target_dim, name="final_projection")
        self.blocks = [
            DecoderLayer(
                hidden_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                ffn_intermediate_size,
                hidden_dropout_prob,
                layer_norm_eps,
            )
            for _ in range(num_decoder_layers)
        ]

    def embed(self, values, offset=0):
        x = self.embedding(values)
        position = tf.cast(tf.range(tf.shape(values)[1]) + offset, tf.float32)[:, None]
        dimension = tf.range(self.hidden_size)
        angles = position / tf.pow(10000.0, tf.cast(2 * (dimension // 2), tf.float32) / self.hidden_size)
        encoding = tf.where(dimension % 2 == 0, tf.sin(angles), tf.cos(angles))
        return x + tf.cast(encoding[None, ...], x.dtype)

    def initialize_state(self, features, horizon):
        batch = tf.shape(features)[0]
        if not self.use_cache:
            return tf.zeros([batch, horizon, self.target_dim + features.shape[-1]], features.dtype)
        return tuple(
            (
                tf.zeros([batch, horizon, self.hidden_size], features.dtype),
                tf.zeros([batch, horizon, self.hidden_size], features.dtype),
            )
            for _ in self.blocks
        )

    def sequence(self, previous, features, memory, training=None):
        x = self.embed(tf.concat([previous, features], axis=-1))
        for block in self.blocks:
            x, _ = block(x, memory, training=training)
        return self.projection(x)

    def step(self, previous, state, features, memory, *, offset, training=None):
        values = tf.concat([previous, features], axis=-1)
        if not self.use_cache:
            slot = tf.one_hot(offset, tf.shape(state)[1], dtype=state.dtype)[None, :, None]
            state = state * (1 - slot) + values * slot
            x = self.embed(state[:, : offset + 1, :])
            for block in self.blocks:
                x, _ = block(x, memory, training=training)
            return StepOutput(self.projection(x[:, -1:, :]), state=state)
        x = self.embed(values, offset)
        caches = []
        for block, cache in zip(self.blocks, state):
            x, cache = block(x, memory, cache=cache, offset=offset, training=training)
            caches.append(cache)
        return StepOutput(self.projection(x), state=tuple(caches))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                name: getattr(self, name)
                for name in (
                    "predict_sequence_length",
                    "num_decoder_layers",
                    "hidden_size",
                    "num_attention_heads",
                    "attention_probs_dropout_prob",
                    "ffn_intermediate_size",
                    "hidden_dropout_prob",
                    "layer_norm_eps",
                    "target_dim",
                    "use_cache",
                )
            }
        )
        return config


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
