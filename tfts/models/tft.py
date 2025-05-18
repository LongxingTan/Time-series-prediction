"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from ..layers.attention_layer import Attention, SelfAttention
from ..layers.dense_layer import FeedForwardNetwork
from ..layers.embed_layer import DataEmbedding
from .base import BaseConfig, BaseModel


class TFTransformerConfig(BaseConfig):
    model_type: str = "tft"

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        ffn_intermediate_size: int = 256,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs
    ):
        super(TFTransformerConfig, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.update(kwargs)


class TFTransformer(BaseModel):
    """Temporal fusion transformer model"""

    def __init__(self, predict_sequence_length=1, config: Optional[TFTransformerConfig] = None):
        super(TFTransformer, self).__init__()
        self.config = config or TFTransformerConfig()
        self.predict_sequence_length = predict_sequence_length

        # Embedding layers for temporal and static features
        self.temporal_embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")
        self.static_embedding = DataEmbedding(self.config.hidden_size)

        # Variable selection networks (simplified as dense layers with gating)
        self.temporal_variable_selection = Dense(self.config.hidden_size, activation="sigmoid")
        self.static_variable_selection = Dense(self.config.hidden_size, activation="sigmoid")

        # Gated Residual Networks (GRN) for feature processing
        self.temporal_grn = FeedForwardNetwork(
            self.config.hidden_size, self.config.ffn_intermediate_size, self.config.hidden_dropout_prob
        )
        self.static_grn = FeedForwardNetwork(
            self.config.hidden_size, self.config.ffn_intermediate_size, self.config.hidden_dropout_prob
        )

        # Static covariate encoder (using LSTM)
        self.static_encoder = tf.keras.layers.LSTM(self.config.hidden_size, return_sequences=True)

        # Temporal fusion decoder (combining LSTM, attention, and gating)
        self.temporal_decoder = tf.keras.layers.LSTM(self.config.hidden_size, return_sequences=True)
        self.attention = Attention(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )
        self.gate = Dense(self.config.hidden_size, activation="sigmoid")

        # Output projection
        self.output_projection = Dense(1)

    def __call__(self, x: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None):
        """Process inputs through the TFT model.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, sequence_length, features).
        output_hidden_states : bool, optional
            Whether to output hidden states, by default None.
        return_dict : bool, optional
            Whether to return a dictionary of outputs, by default None.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, predict_sequence_length, 1).
        """
        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        # Embed temporal and static features
        temporal_embedded = self.temporal_embedding(encoder_feature)
        static_embedded = self.static_embedding(decoder_feature)

        # Apply variable selection
        temporal_selected = self.temporal_variable_selection(temporal_embedded)
        static_selected = self.static_variable_selection(static_embedded)

        # Process through Gated Residual Networks
        temporal_processed = self.temporal_grn(temporal_selected)
        static_processed = self.static_grn(static_selected)

        # Encode static covariates
        static_encoded = self.static_encoder(static_processed)

        # Decode temporal features
        temporal_decoded = self.temporal_decoder(temporal_processed)

        # Apply attention and gating
        attention_output = self.attention(temporal_decoded, static_encoded, static_encoded)
        gate_output = self.gate(attention_output)
        fused_output = gate_output * attention_output

        # Project to output
        output = self.output_projection(fused_output)

        # Slice the output to only include the last predict_sequence_length steps
        output = output[:, -self.predict_sequence_length :, :]

        return output
