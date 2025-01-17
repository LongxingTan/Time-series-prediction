"""
`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
<https://arxiv.org/abs/1810.04805>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling1D, BatchNormalization, Dense, Dropout, LayerNormalization

from tfts.layers.attention_layer import Attention, SelfAttention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding, TokenEmbedding, TokenRnnEmbedding
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
        self.classifier_dropout: Optional[float] = classifier_dropout
        self.pad_token_id: int = pad_token_id


class Bert(BaseModel):
    """Bert model for time series"""

    def __init__(self, predict_sequence_length: int = 1, config=None) -> None:
        super(Bert, self).__init__()
        if config is None:
            config = BertConfig()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        self.encoder_embedding = TokenEmbedding(config.hidden_size)
        self.encoder = Encoder(
            num_hidden_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            ffn_intermediate_size=config.ffn_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.dense1 = Dense(512, activation="relu")
        self.dense2 = Dense(1024, activation="relu")
        self.project1 = Dense(predict_sequence_length, activation=None)

        # self.forecasting = Forecasting(predict_sequence_length, self.config)
        # self.pool1 = AveragePooling1D(pool_size=6)
        # self.rnn1 = GRU(units=1, activation='tanh', return_state=False, return_sequences=True, dropout=0)
        # self.rnn2 = GRU(units=32, activation='tanh', return_state=False, return_sequences=True, dropout=0)
        # self.project2 = Dense(48, activation=None)
        # self.project3 = Dense(1, activation=None)
        #
        # self.dense_se = Dense(16, activation='relu')
        # self.dense_se2 = Dense(1, activation='sigmoid')

    def __call__(
        self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None, return_dict: Optional[bool] = None
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
        # encoder_features = self.spatial_drop(encoder_features)
        # encoder_features_res = self.tcn(encoder_features)
        # encoder_features += encoder_features_res

        # batch * train_sequence * (hidden * heads)
        memory = self.encoder(encoder_feature, encoder_mask=None)
        encoder_output = memory[:, -1]

        # encoder_output = self.bn1(encoder_output)
        # encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.bn2(encoder_output)
        # encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        # encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)
        # outputs = tf.repeat(outputs, [6]*48, axis=1)

        # se = self.dense_se(decoder_features)  # batch * pred_len * 1
        # se = self.dense_se2(se)
        # outputs = tf.math.multiply(outputs, se)

        # memory
        # x2 = self.rnn1(memory)
        # outputs += 0.2 * x2

        # outputs2 = self.project2(encoder_output)  # 48
        # outputs2 = tf.repeat(outputs2, repeats=[6]*48, axis=1)
        # outputs2 = tf.expand_dims(outputs2, -1)
        # outputs += outputs2

        # outputs = self.forecasting(encoder_features, teacher)
        # outputs = tf.math.cumsum(outputs, axis=1)

        # grafting
        # base = decoder_features[:, :, -1:]
        # outputs += base

        return outputs
