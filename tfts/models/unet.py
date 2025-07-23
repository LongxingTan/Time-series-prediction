"""
`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_
"""

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling1D,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Lambda,
    LayerNormalization,
    MultiHeadAttention,
    UpSampling1D,
)

from tfts.layers.embed_layer import DataEmbedding
from tfts.layers.unet_layer import ConvbrLayer, ReBlock, SeBlock

from ..layers.util_layer import ShapeLayer
from .base import BaseConfig, BaseModel


class UnetConfig(BaseConfig):
    model_type: str = "unet"

    def __init__(
        self,
        units: int = 64,
        kernel_size: int = 2,
        depth: int = 2,
        pool_sizes: Tuple[int, int] = (2, 4),
        upsampling_factors: Tuple[int, int, int] = (2, 2, 2),
        num_attention_heads: int = 4,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        use_residual: bool = False,
        use_attention: bool = False,
        use_se: bool = False,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super(UnetConfig, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.depth = depth
        self.pool_sizes = pool_sizes
        self.upsampling_factors = upsampling_factors
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_se = use_se
        self.use_layer_norm = use_layer_norm
        self.update(kwargs)


class Unet(BaseModel):
    """Unet model for sequence-to-sequence prediction tasks."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[UnetConfig] = None):
        super(Unet, self).__init__()
        self.config = config or UnetConfig()
        self.predict_sequence_length = predict_sequence_length

        # Validate sequence length requirements
        min_sequence_length = (
            self.config.pool_sizes[0]
            * self.config.pool_sizes[1]
            * self.config.upsampling_factors[0]
            * self.config.upsampling_factors[1]
            * self.config.upsampling_factors[2]
        )
        if predict_sequence_length > min_sequence_length:
            raise ValueError(
                f"predict_sequence_length ({predict_sequence_length}) must be less than or equal to "
                f"the minimum sequence length ({min_sequence_length}) determined by pooling and upsampling factors. "
                f"Current pool_sizes={self.config.pool_sizes}, upsampling_factors={self.config.upsampling_factors}"
            )

        # Input embedding
        self.embedding = DataEmbedding(self.config.units, positional_type="positional encoding")

        # Pooling layers
        self.avg_pool1 = AveragePooling1D(pool_size=self.config.pool_sizes[0])
        self.avg_pool2 = AveragePooling1D(pool_size=self.config.pool_sizes[1])

        # Encoder and decoder
        self.encoder = Encoder(
            units=self.config.units,
            kernel_size=self.config.kernel_size,
            depth=self.config.depth,
            use_attention=self.config.use_attention,
            use_residual=self.config.use_residual,
            use_se=self.config.use_se,
            use_layer_norm=self.config.use_layer_norm,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
        )

        self.decoder = Decoder(
            upsampling_factors=self.config.upsampling_factors,
            units=self.config.units,
            kernel_size=self.config.kernel_size,
            predict_seq_length=predict_sequence_length,
            use_attention=self.config.use_attention,
            use_residual=self.config.use_residual,
            use_se=self.config.use_se,
            use_layer_norm=self.config.use_layer_norm,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
        )

        # Output projection
        self.output_projection = Dense(1)

    def __call__(
        self,
        x: tf.Tensor,
        training: bool = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).
            training: Boolean flag for training mode.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary of outputs.

        Returns:
            Tensor: Output predictions of shape (batch_size, predict_sequence_length, 1).
        """
        # Validate input sequence length
        # _, input_sequence_length, _ = ShapeLayer()(x)
        # min_sequence_length = (
        #     self.config.pool_sizes[0]
        #     * self.config.pool_sizes[1]
        #     * self.config.upsampling_factors[0]
        #     * self.config.upsampling_factors[1]
        #     * self.config.upsampling_factors[2]
        # )
        # if input_sequence_length < min_sequence_length:
        #     raise ValueError(
        #         f"Input sequence length ({input_sequence_length}) must be greater than or equal to "
        #         f"the minimum sequence length ({min_sequence_length}) determined by pooling and upsampling factors. "
        #         f"Current pool_sizes={self.config.pool_sizes}, upsampling_factors={self.config.upsampling_factors}"
        #     )

        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        # Embed inputs
        x = self.embedding(encoder_feature)

        # Apply pooling
        pool1 = self.avg_pool1(x)
        pool2 = self.avg_pool2(x)

        # Encode
        encoder_output = self.encoder([x, pool1, pool2], training=training)

        # Decode
        decoder_output = self.decoder(encoder_output, training=training)

        # Project to output
        output = self.output_projection(decoder_output)

        # Slice to prediction length
        output = output[:, -self.predict_sequence_length :, :]

        if return_dict:
            return {"output": output}
        return output


class Encoder(tf.keras.layers.Layer):
    """Encoder component for the Unet model."""

    def __init__(
        self,
        units: int = 64,
        kernel_size: int = 2,
        depth: int = 1,
        use_attention: bool = False,
        use_residual: bool = False,
        use_se: bool = False,
        use_layer_norm: bool = False,
        num_attention_heads: int = 4,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.depth = depth
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_se = use_se
        self.use_layer_norm = use_layer_norm

        # First level layers
        self.conv_br1 = ConvbrLayer(units, kernel_size, 1, 1)
        self.re_blocks1 = [ReBlock(units, kernel_size, 1, 1, use_se=use_se) for _ in range(depth)]

        # Second level layers
        self.conv_br2 = ConvbrLayer(units * 2, kernel_size, 2, 1)
        self.re_blocks2 = [ReBlock(units * 2, kernel_size, 1, 1, use_se=use_se) for _ in range(depth)]

        # Third level layers
        self.conv_br3 = ConvbrLayer(units * 3, kernel_size, 2, 1)
        self.re_blocks3 = [ReBlock(units * 3, kernel_size, 1, 1, use_se=use_se) for _ in range(depth)]

        # Fourth level layers
        self.conv_br4 = ConvbrLayer(units * 4, kernel_size, 2, 1)
        self.re_blocks4 = [ReBlock(units * 4, kernel_size, 1, 1, use_se=use_se) for _ in range(depth)]

        # Attention layers
        if use_attention:
            self.attention_layers = [
                MultiHeadAttention(num_heads=num_attention_heads, key_dim=units, dropout=attention_probs_dropout_prob)
                for _ in range(depth)
            ]

        # Layer normalization
        if use_layer_norm:
            self.layer_norms = [LayerNormalization() for _ in range(depth)]

        # Dropout
        self.dropout = Dropout(hidden_dropout_prob)

    def call(self, inputs: tf.Tensor, training: bool = True):
        """Forward pass through the encoder.

        Args:
            inputs: Tuple containing the input tensor and pooled tensors.
            training: Whether the model is in training mode.

        Returns:
            Tuple: Encoder outputs.
        """
        x, pool1, pool2 = inputs

        # First level
        x = self.conv_br1(x)  # => batch_size * sequence_length * units
        for i in range(self.depth):
            residual = x
            x = self.re_blocks1[i](x)
            if self.use_attention:
                x = self.attention_layers[i](x, x, x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            if self.use_residual:
                x = x + residual
            x = self.dropout(x, training=training)
        out_0 = x  # => batch_size * sequence_length * units

        # Second level
        x = self.conv_br2(x)

        for i in range(self.depth):
            residual = x
            x = self.re_blocks2[i](x)
            if self.use_attention:
                x = self.attention_layers[i](x, x, x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
        out_1 = x  # => batch_size * (sequence/2) * (units * 2)

        # Third level with pool1
        x = Concatenate()([x, pool1])
        x = self.conv_br3(x)
        for i in range(self.depth):
            residual = x
            x = self.re_blocks3[i](x)
            if self.use_attention:
                x = self.attention_layers[i](x, x, x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            if self.use_residual:
                x = x + residual
        out_2 = x  # => batch_size * (sequence/2), (units*3)

        # Fourth level with pool2
        x = Concatenate()([x, pool2])
        x = self.conv_br4(x)
        for i in range(self.depth):
            residual = x
            x = self.re_blocks4[i](x)
            if self.use_attention:
                x = self.attention_layers[i](x, x, x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            if self.use_residual:
                x = x + residual
            x = self.dropout(x, training=training)
        out3 = x
        return [out_0, out_1, out_2, out3]


class Decoder(tf.keras.layers.Layer):
    """Decoder component for the Unet model."""

    def __init__(
        self,
        upsampling_factors: Tuple[int, int, int],
        units: int = 64,
        kernel_size: int = 2,
        predict_seq_length: int = 1,
        use_attention: bool = True,
        use_residual: bool = True,
        use_se: bool = True,
        use_layer_norm: bool = True,
        num_attention_heads: int = 4,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.upsampling_factors = upsampling_factors
        self.units = units
        self.kernel_size = kernel_size
        self.predict_seq_length = predict_seq_length
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_se = use_se
        self.use_layer_norm = use_layer_norm

        # Upsampling layers
        self.upsampling1 = UpSampling1D(upsampling_factors[0])
        self.upsampling2 = UpSampling1D(upsampling_factors[1])
        self.upsampling3 = UpSampling1D(upsampling_factors[2])

        # Convolution layers
        self.conv_br1 = ConvbrLayer(units * 3, kernel_size, 1, 1)
        self.conv_br2 = ConvbrLayer(units * 2, kernel_size, 1, 1)
        self.conv_br3 = ConvbrLayer(units, kernel_size, 1, 1)

        # Attention layers
        if use_attention:
            self.attention_layers = [
                MultiHeadAttention(num_heads=num_attention_heads, key_dim=units, dropout=attention_probs_dropout_prob)
                for _ in range(3)  # One for each upsampling level
            ]

        # Layer normalization
        if use_layer_norm:
            self.layer_norms = [LayerNormalization() for _ in range(3)]

        # Dropout
        self.dropout = Dropout(hidden_dropout_prob)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], training: bool = True) -> tf.Tensor:
        """Forward pass through the decoder.

        Args:
            inputs: Tuple containing encoder outputs.
            training: Whether the model is in training mode.

        Returns:
            Tensor: Decoder output.
        """
        out_0, out_1, out_2, x = inputs

        # First upsampling
        x = self.upsampling1(x)
        x = Concatenate()([x, out_2])
        x = self.conv_br1(x)
        if self.use_attention:
            x = self.attention_layers[0](x, x, x)
        if self.use_layer_norm:
            x = self.layer_norms[0](x)
        x = self.dropout(x, training=training)

        # Second upsampling
        x = self.upsampling2(x)
        x = Concatenate()([x, out_1])
        x = self.conv_br2(x)
        if self.use_attention:
            x = self.attention_layers[1](x, x, x)
        if self.use_layer_norm:
            x = self.layer_norms[1](x)
        x = self.dropout(x, training=training)

        # Third upsampling
        x = self.upsampling3(x)
        x = Concatenate()([x, out_0])
        x = self.conv_br3(x)
        if self.use_attention:
            x = self.attention_layers[2](x, x, x)
        if self.use_layer_norm:
            x = self.layer_norms[2](x)
        x = self.dropout(x, training=training)
        return x
