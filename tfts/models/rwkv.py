"""
`RWKV: Reinventing RNNs for the Transformer Era
<https://arxiv.org/abs/2305.13048>`_
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import GRU, BatchNormalization, Dense, Dropout

from tfts.layers.rwkv_layer import ChannelMixing, TimeMixing

from .base import BaseConfig, BaseModel


class RWKVConfig(BaseConfig):
    model_type: str = "rwkv"

    def __init__(
        self,
        num_layers: int = 25,
        hidden_size: int = 64,
        vocab_size: int = 50304,
        dense_hidden_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the configuration for the RWKV model with the specified parameters.

        Args:
            num_layers: The number of stacked RWKV layers.
            hidden_size: Size of each attention head.
            dense_hidden_size: The size of the dense hidden layer following the RWKV.
        """
        super().__init__()

        self.vocab_size: int = vocab_size
        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        self.dense_hidden_size: int = dense_hidden_size
        self.dropout: float = dropout


class RWKV(BaseModel):
    """TensorFlow RWKV model"""

    def __init__(self, predict_sequence_length: int = 1, config=None):
        super().__init__(config)
        if config is None:
            config = RWKVConfig()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        self.emb = tf.keras.layers.Embedding(config.vocab_size, config.n_embd)
        self.ln0 = tf.keras.layers.LayerNormalization()

        self.blocks = [RWKVBlock(config) for _ in range(config.n_layer)]

        self.ln_out = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

    def init_state(self, batch_size=1):
        states = []
        for _ in range(self.config.n_layer):
            # States for attention
            att_states = [
                tf.zeros((batch_size, self.config.n_embd)),  # aa
                tf.zeros((batch_size, self.config.n_embd)),  # bb
                tf.zeros((batch_size, self.config.n_embd)) - 1e30,  # pp
            ]
            # State for FFN
            ffn_state = tf.zeros((batch_size, self.config.n_embd))
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
        """RWKV model call"""

        if states is None:
            states = self.init_state()

        x = self.emb(x)
        x = self.ln0(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            x, block_states = block(x, states[i])
            new_states.append(block_states)

        x = self.ln_out(x)
        x = self.head(x)

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


class RWKVBlock(tf.keras.layers.Layer):
    """TensorFlow RWKV block"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.attention = TimeMixing(config)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.feed_forward = ChannelMixing(config)

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
