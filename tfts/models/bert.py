"""
`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
<https://arxiv.org/abs/1810.04805>`_
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape

from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding
from tfts.models.transformer import Encoder

from .base import BaseConfig, BaseModel


class BertConfig(BaseConfig):

    model_type: str = "bert"

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
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        dense_units: Tuple[int] = (512, 1024),
        classifier_dropout: Optional[float] = None,
        **kwargs: Dict[str, object]
    ) -> None:
        """Configuration class for BERT model, inheriting from BaseConfig.

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
            position_embedding_type: The type of position embedding ("absolute" or "relative"). Default is "absolute".
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
        self.position_embedding_type: str = position_embedding_type
        self.use_cache: bool = use_cache
        self.dense_unites: Tuple[int] = dense_units
        self.classifier_dropout: Optional[float] = classifier_dropout
        self.pad_token_id: int = pad_token_id


class Bert(BaseModel):
    """Bert model for time series"""

    def __init__(self, predict_sequence_length: int = 1, config=None) -> None:
        super(Bert, self).__init__()
        self.config = config or BertConfig()
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
        """Bert model call

        Parameters
        ----------
        inputs : tf.Tensor
            BERT model input
        teacher : tf.Tensor, optional
            teacher forcing for autoregression, by default None

        Returns
        -------
        tf.Tensor
            BERT model output tensor as prediction output
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
