"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, LayerNormalization

from ..layers.attention_layer import Attention, SelfAttention
from ..layers.dense_layer import FeedForwardNetwork
from ..layers.embed_layer import DataEmbedding
from .base import BaseConfig, BaseModel


class TFTransformerConfig(BaseConfig):
    model_type: str = "tft"

    def __init__(
        self,
        encoder_input_dim: int = 1,
        decoder_input_dim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        output_size: int = 1,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        ffn_intermediate_size: int = 256,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super(TFTransformerConfig, self).__init__()
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.output_size = output_size
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

        # Variable selection networks
        self.encoder_var_selection = tf.keras.Sequential(
            [
                Dense(self.config.hidden_size, activation="relu"),
                Dense(self.config.encoder_input_dim, activation="sigmoid"),
            ],
            name="encoder_var_selection",
        )
        self.decoder_var_selection = tf.keras.Sequential(
            [
                Dense(self.config.hidden_size, activation="relu"),
                Dense(self.config.decoder_input_dim, activation="sigmoid"),
            ],
            name="decoder_var_selection",
        )

        self.lstm_encoder_layers = [
            LSTM(
                self.config.hidden_size,
                return_sequences=True,
                dropout=0.0 if i < self.config.num_layers - 1 else 0.0,
                name=f"lstm_enc_{i}",
            )
            for i in range(self.config.num_layers)
        ]
        self.lstm_decoder_layers = [
            LSTM(
                self.config.hidden_size,
                return_sequences=True,
                dropout=0.0 if i < self.config.num_layers - 1 else 0.0,
                name=f"lstm_dec_{i}",
            )
            for i in range(self.config.num_layers)
        ]

        self.attention = Attention(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )
        self.concat = Concatenate(axis=1)

        # Output projection
        self.output_projection = Dense(self.config.output_size)

    def __call__(
        self,
        x: Optional[tf.Tensor] = None,
        encoder_cat=None,
        encoder_num=None,
        decoder_cat=None,
        decoder_num=None,
        static_cat=None,
        static_num=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
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

        # encoder_input = tf.concat([encoder_num, encoder_cat], axis=2)
        # decoder_input = tf.concat([decoder_num, decoder_cat], axis=2)

        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        encoder_weights = self.encoder_var_selection(encoder_feature)
        encoder_feature = encoder_feature * encoder_weights

        decoder_weights = self.decoder_var_selection(decoder_feature)
        decoder_feature = decoder_feature * decoder_weights

        # Encode static covariates
        temporal_encoded = encoder_feature
        for layer in self.lstm_encoder_layers:
            temporal_encoded = layer(temporal_encoded)

        # Decode temporal features
        temporal_decoded = temporal_encoded
        for layer in self.lstm_decoder_layers:
            temporal_decoded = layer(temporal_decoded)

        sequence = self.concat([temporal_encoded, temporal_decoded])
        attention_output = self.attention(sequence, sequence, sequence)

        attention_output = attention_output[:, -self.predict_sequence_length :, :]
        output = self.output_projection(attention_output)

        return output
