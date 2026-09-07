"""
`Scalable Diffusion Models with Transformers
<https://arxiv.org/abs/2212.09748>`_
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization

from tfts.layers.attention_layer import Attention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding

from .base import BaseModel, CommonConfig
from .registry import register_model


class DiffusionConfig(CommonConfig):
    model_type: str = "diffusion"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        ffn_intermediate_size: int = 256,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        **kwargs,
    ) -> None:
        """
        Initializes the configuration for the Diffusion model with the specified parameters.

        Args:
            hidden_size: Size of each attention head.
            num_layers: The number of stacked transformer layers.
            num_attention_heads: The number of attention heads.
            attention_probs_dropout_prob: Dropout rate for attention probabilities.
            hidden_dropout_prob: Dropout rate for hidden layers.
            ffn_intermediate_size: Size of the intermediate layer in the feed-forward network.
            max_position_embeddings: Maximum sequence length for positional embeddings.
            initializer_range: Standard deviation for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            pad_token_id: ID for padding token.
            num_diffusion_steps: Number of diffusion steps.
            beta_start: Starting noise level.
            beta_end: Ending noise level.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_attention_heads: int = num_attention_heads
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob
        self.hidden_dropout_prob: float = hidden_dropout_prob
        self.ffn_intermediate_size: int = ffn_intermediate_size
        self.max_position_embeddings: int = max_position_embeddings
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.pad_token_id: int = pad_token_id
        self.num_diffusion_steps: int = num_diffusion_steps
        self.beta_start: float = beta_start
        self.beta_end: float = beta_end
        self.update(kwargs)


class NoiseScheduler:
    """Linear noise scheduler for diffusion models"""

    def __init__(self, config: DiffusionConfig):
        self.num_diffusion_steps = config.num_diffusion_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end

        # Create linear schedule
        self.betas = tf.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x: tf.Tensor, t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Add noise to the input at timestep t"""
        noise = tf.random.normal(shape=tf.shape(x))
        alpha_t = tf.gather(self.sqrt_alphas_cumprod, t)
        alpha_t = tf.reshape(alpha_t, [-1, 1, 1])
        beta_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        beta_t = tf.reshape(beta_t, [-1, 1, 1])

        noisy_x = alpha_t * x + beta_t * noise
        return noisy_x, noise

    def remove_noise(self, x: tf.Tensor, noise: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Remove noise from the input at timestep t"""
        alpha_t = tf.gather(self.sqrt_alphas_cumprod, t)
        alpha_t = tf.reshape(alpha_t, [-1, 1, 1])
        beta_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        beta_t = tf.reshape(beta_t, [-1, 1, 1])

        denoised_x = (x - beta_t * noise) / alpha_t
        return denoised_x


@register_model(
    "diffusion",
    config=DiffusionConfig,
    paper="https://arxiv.org/abs/2006.11239",
    tags=("generative", "diffusion", "probabilistic"),
)
class Diffusion(BaseModel):
    """TensorFlow Diffusion model for time series forecasting"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[DiffusionConfig] = None):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.predict_sequence_length = predict_sequence_length
        self.noise_scheduler = NoiseScheduler(self.config)

        # Layers that don't depend on input feature count
        self.time_embedding = Dense(self.config.hidden_size)
        self.embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")
        self.blocks = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]

        # Kept for compatibility with callers that inspect this layer.
        self.output_projection = Dense(1)

        # The forecast head is sized for the input channel count in build().
        # (The base implementation returned a reconstruction of the *input tail*,
        #  which is not a forecast of the held-out future; this head fixes that.)
        self.forecast_projection = None

    def build(self, input_shape):
        """Create a channel-aware forecast projection from the input contract."""
        _, encoder_shape = self._input_shapes(input_shape)
        channels = encoder_shape[-1]
        if channels is None:
            raise ValueError("Diffusion requires a statically known input channel count")
        self.forecast_projection = Dense(self.predict_sequence_length * int(channels))
        super().build(input_shape)

    def call(self, x, training=None, **kwargs):
        """Diffusion model forward pass logic."""
        # 1. Prepare inputs (using BaseModel helper)
        # Note: ignore_decoder_inputs=True because diffusion usually denoises the encoder path
        x, encoder_feature, _ = self._prepare_3d_inputs(x, ignore_decoder_inputs=True)

        # 2. Generate random timesteps (training only).
        batch_size = tf.shape(encoder_feature)[0]
        if training:
            t = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.config.num_diffusion_steps, dtype=tf.int32)
            # 3. Add noise to input (diffusion forward process) during training.
            noisy_x, _ = self.noise_scheduler.add_noise(encoder_feature, t)
            t_float = tf.cast(t, tf.float32)
        else:
            # Inference: condition on the CLEAN history at t=0 (no noise), matching the
            # standard diffusion-forecasting protocol. Random noise at inference produced
            # degenerate, stochastic forecasts (near-constant output).
            noisy_x = encoder_feature
            t_float = tf.zeros([batch_size], dtype=tf.float32)

        # 4. Time embedding (batch, 1) -> (batch, 1, hidden)
        t_emb = self.time_embedding(tf.expand_dims(t_float, axis=-1))
        t_emb = tf.expand_dims(t_emb, axis=1)

        # 5. Transformer process
        x = self.embedding(noisy_x)
        x = x + t_emb  # Inject time information

        for block in self.blocks:
            x = block(x)

        # 6. Forecast the future window from the contextualized encoding.
        forecast = self.forecast_projection(x)  # (batch, seq, pred * channels)
        # Use the *last* (most recent) history token's forecast; mean-pooling over all
        # 24 time steps smoothed the output toward a constant and destroyed dynamic range.
        forecast = forecast[:, -1, :]  # (batch, pred * channels)
        channels = tf.shape(encoder_feature)[-1]
        return tf.reshape(forecast, [tf.shape(forecast)[0], self.predict_sequence_length, channels])


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block for Diffusion model"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )
        self.attention_output = Dense(config.hidden_size)
        self.attention_norm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.attention_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.feed_forward = FeedForwardNetwork(
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.feed_forward_norm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.feed_forward_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, x):
        """Transformer block forward pass"""
        # Self-attention
        attention_output = self.attention(x, x, x)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        x = self.attention_norm(x + attention_output)

        # Feed-forward
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        x = self.feed_forward_norm(x + feed_forward_output)

        return x
