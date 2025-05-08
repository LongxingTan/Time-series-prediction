"""
`Language Models are Few-Shot Learners
<https://arxiv.org/abs/2005.14165>`_
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape

from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding
from tfts.models.transformer import Encoder

from .base import BaseConfig, BaseModel


class GPTConfig(BaseConfig):

    model_type: str = "gpt"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        ffn_intermediate_size: int = 256,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        positional_type: str = "absolute",
        use_cache: bool = True,
        dense_units: Tuple[int] = (512, 1024),
        classifier_dropout: Optional[float] = None,
        **kwargs: Dict[str, object],
    ) -> None:
        """Configuration class for GPT decoder model, inheriting from BaseConfig.

        Args:
            hidden_size: The size of the hidden layers. Default is 64.
            num_hidden_layers: The number of hidden layers in the transformer encoder. Default is 2.
            num_attention_heads: The number of attention heads in each attention layer. Default is 4.
            ffn_intermediate_size: The size of the intermediate (feed-forward) layer. Default is 256.
            hidden_act: The activation function for hidden layers. Default is "gelu".
            hidden_dropout_prob: The dropout probability for hidden layers. Default is 0.1.
            attention_probs_dropout_prob: The dropout probability for attention probabilities. Default is 0.1.
            max_position_embeddings: The maximum length of the input sequences. Default is 512.
            type_vocab_size: The vocabulary size for token types (usually 2). Default is 2.
            initializer_range: The standard deviation for weight initialization. Default is 0.02.
            layer_norm_eps: The epsilon value for layer normalization. Default is 1e-12.
            pad_token_id: The ID for the padding token. Default is 0.
            positional_type: The type of position embedding ("absolute" or "relative"). Default is "absolute".
            use_cache: Whether to use the cache during inference. Default is True.
            classifier_dropout: Dropout probability for the classifier layer. Default is None.
            **kwargs: Additional keyword arguments passed to the parent `BaseConfig` class.
        """

        super().__init__(**kwargs)

        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_attention_heads: int = num_attention_heads
        self.ffn_intermediate_size: int = ffn_intermediate_size
        self.hidden_act: str = hidden_act
        self.hidden_dropout_prob: float = hidden_dropout_prob
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob
        self.max_position_embeddings: int = max_position_embeddings
        self.type_vocab_size: int = type_vocab_size
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.positional_type: str = positional_type
        self.use_cache: bool = use_cache
        self.dense_unites: Tuple[int] = dense_units
        self.classifier_dropout: Optional[float] = classifier_dropout
        self.pad_token_id: int = pad_token_id

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if not 0 <= self.attention_probs_dropout_prob < 1:
            raise ValueError(f"attention_probs_dropout_prob must be in [0, 1), got {self.attention_probs_dropout_prob}")
        if not 0 <= self.hidden_dropout_prob < 1:
            raise ValueError(f"hidden_dropout_prob must be in [0, 1), got {self.hidden_dropout_prob}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by attention_heads, got {self.hidden_size}/{self.num_attention_heads}"
            )


class GPT(BaseModel):
    """GPT decoder model for time series"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[GPTConfig] = None) -> None:
        super(GPT, self).__init__()
        self.config = config or GPTConfig()
        self.predict_sequence_length = predict_sequence_length

        self.encoder_embedding = TokenEmbedding(self.config.hidden_size)
        self.encoder = Encoder(
            num_hidden_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
        )

        self.dense_layers = []
        for unit in self.config.dense_unites:
            self.dense_layers.append(Dense(unit, activation="relu"))

        self.projection = Dense(predict_sequence_length, activation=None)

    def __call__(
        self,
        inputs: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tf.Tensor:
        """GPT model forward pass.

        Args:
            inputs: Input time-series data, can be:
                - A single tensor of shape [batch_size, seq_len, feature_dim]
                - A tuple/list of (x, encoder_feature, decoder_feature)
                - A dictionary with keys 'x' and 'encoder_feature'
            teacher: Optional teacher forcing tensor for autoregression.
            training: Whether the model is in training mode.
            mask: Optional attention mask.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return outputs as a dictionary.

        Returns:
            If return_dict is False and output_hidden_states is False:
                Forecasted values tensor of shape [batch_size, predict_sequence_length, input_dim]
            If output_hidden_states is True:
                Hidden states from the encoder.
            If return_dict is True:
                Dictionary containing model outputs.
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs

        encoder_feature = self.encoder_embedding(encoder_feature)

        memory = self.encoder(encoder_feature, mask=None)

        if output_hidden_states:
            # (batch_size, train_sequence_length, hidden_size)
            return memory

        encoder_output = memory[:, -1]

        for layer in self.dense_layers:
            encoder_output = layer(encoder_output)
        outputs = self.projection(encoder_output)
        outputs = Reshape((outputs.shape[1], 1))(outputs)
        return outputs
