"""
`RWKV: Reinventing RNNs for the Transformer Era
<https://arxiv.org/abs/2305.13048>`_
"""

from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense

from tfts.layers.embed_layer import DataEmbedding
from tfts.layers.rwkv_layer import ChannelMixing, TimeMixing

from .base import BaseConfig, BaseModel


class RWKVConfig(BaseConfig):
    model_type: str = "rwkv"

    def __init__(
        self,
        num_layers: int = 25,
        hidden_size: int = 64,
        dense_hidden_size: int = 32,
        dropout: float = 0.0,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs
    ) -> None:
        """
        Initializes the configuration for the RWKV model with the specified parameters.

        Args:
            num_layers: The number of stacked RWKV layers.
            hidden_size: Size of each attention head.
            dense_hidden_size: The size of the dense hidden layer following the RWKV.
            dropout: Dropout rate for regularization.
            max_position_embeddings: Maximum sequence length for positional embeddings.
            initializer_range: Standard deviation for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            pad_token_id: ID for padding token.
        """
        super().__init__()

        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        self.dense_hidden_size: int = dense_hidden_size
        self.dropout: float = dropout
        self.max_position_embeddings: int = max_position_embeddings
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.pad_token_id: int = pad_token_id
        self.update(kwargs)


class RWKV(BaseModel):
    """TensorFlow RWKV model for time series forecasting"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[RWKVConfig] = None):
        super().__init__()
        self.config = config or RWKVConfig()
        self.predict_sequence_length = predict_sequence_length

        # Embedding layer
        self.embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")

        self.ln0 = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps)

        self.blocks = [RWKVBlock(self.config) for _ in range(self.config.num_layers)]

        self.ln_out = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps)
        self.output_projection = Dense(1)

    def init_state(self, batch_size=1):
        states = []
        for _ in range(self.config.num_layers):
            # States for attention
            att_states = [
                tf.zeros((batch_size, self.config.hidden_size)),  # aa
                tf.zeros((batch_size, self.config.hidden_size)),  # bb
                tf.zeros((batch_size, self.config.hidden_size)) - 1e30,  # pp
            ]
            # State for FFN
            ffn_state = tf.zeros((batch_size, self.config.hidden_size))
            states.append((att_states, ffn_state))
        return states

    def __call__(
        self,
        x,
        states=None,
        teacher=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """RWKV model call for time series forecasting"""

        if states is None:
            states = self.init_state()

        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        # Embed inputs
        x = self.embedding(encoder_feature)
        x = self.ln0(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            x, block_states = block(x, states[i])
            new_states.append(block_states)

        x = self.ln_out(x)
        x = self.output_projection(x)

        # Slice the output to only include the last predict_sequence_length steps
        x = x[:, -self.predict_sequence_length :, :]

        return x, new_states

    def _prepare_inputs(self, inputs):
        """Prepare the inputs for the encoder."""

        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, _ = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            x = inputs
            encoder_feature = x
        return x, encoder_feature

    def _prepare_3d_inputs(self, x, ignore_decoder_inputs=True):
        """Prepare the inputs for the encoder and decoder."""

        if isinstance(x, (list, tuple)):
            encoder_feature = x[1] if len(x) > 1 else x[0]
            decoder_feature = x[2] if len(x) > 2 else None
        elif isinstance(x, dict):
            encoder_feature = x["encoder_feature"]
            decoder_feature = x["decoder_feature"] if "decoder_feature" in x else None
        else:
            encoder_feature = x
            decoder_feature = None

        if ignore_decoder_inputs:
            return encoder_feature, encoder_feature, None
        else:
            return encoder_feature, encoder_feature, decoder_feature


class RWKVBlock(tf.keras.layers.Layer):
    """TensorFlow RWKV block"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.attention = TimeMixing(config)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.feed_forward = ChannelMixing(config)

    def build(self, input_shape):
        super().build(input_shape)
        # Ensure the attention and feed_forward layers are built
        self.attention.build(input_shape)
        self.feed_forward.build(input_shape)

    def call(self, x, states):
        """block

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        """
        att_states, ffn_state = states

        # Attention
        h, new_att_states = self.attention(self.ln1(x), att_states)
        x = x + h

        # Feed-forward
        h, new_ffn_state = self.feed_forward(self.ln2(x), ffn_state)
        x = x + h

        return x, (new_att_states, new_ffn_state)
