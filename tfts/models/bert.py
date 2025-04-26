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
        pooling_method: str = "mean",
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        positional_type: str = "positional encoding",
        use_cache: bool = False,
        dense_units: Tuple[int] = (512, 1024),
        classifier_dropout: Optional[float] = None,
        **kwargs: Dict[str, object],
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
        self.pooling_method: str = pooling_method
        self.hidden_act: str = hidden_act
        self.hidden_dropout_prob: float = hidden_dropout_prob
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob
        self.type_vocab_size: int = type_vocab_size
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.positional_type: str = positional_type
        self.use_cache: bool = use_cache
        self.dense_units: Tuple[int] = dense_units
        self.classifier_dropout: Optional[float] = classifier_dropout
        self.pad_token_id: int = pad_token_id


class Bert(BaseModel):
    """Bert model for time series forecasting.

    This model implements a transformer-based architecture (BERT) adapted for time series data.
    It processes time series inputs through a transformer encoder and produces predictions
    for future time steps.

    Parameters
    ----------
    predict_sequence_length : int, optional
        Number of future time steps to predict, by default 1
    config : BertConfig, optional
        Configuration parameters for the model, by default None

    Attributes
    ----------
    config : BertConfig
        Configuration object containing model hyperparameters
    predict_sequence_length : int
        Number of future time steps to predict
    encoder_embedding : DataEmbedding
        Embedding layer for encoder inputs
    encoder : Encoder
        Transformer encoder module
    dense_layers : List[Dense]
        List of dense layers for final projection
    """

    def __init__(self, predict_sequence_length: int = 1, config: Optional[BertConfig] = None) -> None:
        super(Bert, self).__init__()
        self.config = config or BertConfig()
        self.predict_sequence_length = predict_sequence_length
        self.built = False

    def build(self, input_shape):
        """Builds the model layers with the input shape."""

        self.encoder_embedding = DataEmbedding(self.config.hidden_size, positional_type=self.config.positional_type)
        self.encoder = Encoder(
            num_hidden_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
        )
        self.dense_layers = [
            Dense(unit, activation="relu", name=f"dense_{i}") for i, unit in enumerate(self.config.dense_units)
        ]
        self.projection = Dense(self.predict_sequence_length, activation="linear", name="projection")
        self.reshape = Reshape((self.predict_sequence_length, 1))

    def __call__(
        self,
        inputs: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tf.Tensor:
        """Forward pass for Bert model.

        Parameters
        ----------
        inputs : Union[tf.Tensor, Tuple[tf.Tensor, ...], Dict[str, tf.Tensor]]
            Input tensor(s) - can be a single tensor, tuple of tensors, or dictionary
        teacher : tf.Tensor, optional
            Teacher forcing for autoregression, by default None
        output_hidden_states : bool, optional
            Whether to return hidden states, by default False
        return_dict : bool, optional
            Whether to return a dictionary of outputs, by default False

        Returns
        -------
        Union[tf.Tensor, Dict[str, tf.Tensor]]
            Model outputs - either a tensor of predictions or a dictionary of tensors
        """
        if not self.built:
            self.build(inputs.shape)
            self.built = True

        x, encoder_feature, _ = self._prepare_3d_inputs(inputs)

        encoder_feature = self.encoder_embedding(encoder_feature)
        memory = self.encoder(encoder_feature, mask=None)

        if output_hidden_states:
            # (batch_size, train_sequence_length, hidden_size)
            if return_dict:
                return {"hidden_states": memory}
            return memory

        # Extract the mean or last time step for prediction
        if self.config.pooling_method == "mean":
            encoder_output = tf.keras.layers.GlobalAveragePooling1D()(memory)
        elif self.config.pooling_method == "last":
            encoder_output = memory[:, -1]
        else:
            raise ValueError(f"Pooling method should be mean or last, while received {self.config.poolint_method}")

        for layer in self.dense_layers:
            encoder_output = layer(encoder_output)
        outputs = self.projection(encoder_output)
        outputs = self.reshape(outputs)

        if return_dict:
            return {"predictions": outputs}
        return outputs
