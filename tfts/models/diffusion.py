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

from .base import BaseConfig, BaseModel


class DiffusionConfig(BaseConfig):
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
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        num_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        **kwargs
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


class Diffusion(BaseModel):
    """TensorFlow Diffusion model for time series forecasting"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[DiffusionConfig] = None):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.predict_sequence_length = predict_sequence_length
        self.noise_scheduler = NoiseScheduler(self.config)

        # Time embedding
        self.time_embedding = Dense(self.config.hidden_size)

        # Embedding layer
        self.embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")

        # Transformer blocks
        self.blocks = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]

        # Output projection
        self.output_projection = Dense(1)

    def __call__(
        self,
        x,
        states=None,
        teacher=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Diffusion model call for time series forecasting"""
        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        # Generate random timesteps
        batch_size = tf.shape(encoder_feature)[0]
        t = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.config.num_diffusion_steps, dtype=tf.int32)

        # Add noise to input
        noisy_x, noise = self.noise_scheduler.add_noise(encoder_feature, t)

        # Time embedding
        t_emb = self.time_embedding(tf.cast(t, tf.float32))
        t_emb = tf.expand_dims(t_emb, axis=1)

        # Process through transformer blocks
        x = self.embedding(noisy_x)
        x = tf.concat([x, t_emb], axis=-1)

        for block in self.blocks:
            x = block(x)

        # Project to output
        predicted_noise = self.output_projection(x)

        # Remove noise
        denoised_x = self.noise_scheduler.remove_noise(noisy_x, predicted_noise, t)

        # Slice the output to only include the last predict_sequence_length steps
        denoised_x = denoised_x[:, -self.predict_sequence_length :, :]

        return denoised_x


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
