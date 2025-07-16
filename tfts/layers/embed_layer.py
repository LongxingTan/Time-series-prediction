"""Layer for :py:class:`~tfts.models.transformer`"""

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding

from .position_layer import PositionalEmbedding, PositionalEncoding, RelativePositionEmbedding


class DataEmbedding(tf.keras.layers.Layer):
    """
    Data Embedding layer that combines value embeddings with optional positional embeddings.

    This layer first embeds the input data using a TokenEmbedding, then optionally adds
    positional information using one of several positional embedding techniques, and finally
    applies dropout.

    Args:
        embed_size (int): Embedding size for tokens.
        positional_type (str, optional): Type of positional embedding to use.
            Options:
                - "positional encoding": Uses sinusoidal positional encoding
                - "positional embedding": Uses learned positional embeddings
                - "relative encoding": Uses relative position embeddings
                - None: No positional embedding is applied
            Defaults to None.

    Input shape:
        - 3D tensor with shape `(batch_size, time_steps, input_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, time_steps, embed_size)`

    Example:
        ```python
        # Create a DataEmbedding layer with embedding size of 256 and positional encoding
        embedding_layer = DataEmbedding(
            embed_size=256,
            positional_type="positional encoding"
        )

        # Apply to input with shape (batch_size, sequence_length, features)
        input_tensor = tf.random.normal((32, 100, 10))
        output_tensor = embedding_layer(input_tensor)  # Shape: (32, 100, 256)
        ```
    """

    def __init__(self, embed_size: int, positional_type: Optional[str] = "positional encoding", **kwargs):
        """
        Initialize the DataEmbedding layer.

        Args:
            embed_size (int): Embedding size for tokens.
            positional_type (str, optional): Type of positional embedding to use.
                Options: "positional encoding", "positional embedding", "relative encoding", or None.
                Defaults to None.
        """
        super(DataEmbedding, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.positional_type = positional_type

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Value embedding layer: the below section is put in init, so it could build while DataEmbedding is call
        # Otherwise, while load the weights, the TokenEmbedding is not built
        self.value_embedding = TokenEmbedding(self.embed_size)

        # Positional embedding layer based on specified type
        if self.positional_type == "positional encoding":
            self.positional_embedding = PositionalEncoding()
        elif self.positional_type == "positional embedding":
            self.positional_embedding = PositionalEmbedding()
        elif self.positional_type == "relative encoding":
            self.positional_embedding = RelativePositionEmbedding()
        else:
            self.positional_embedding = None
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_length, embed_size).
        """
        # Apply value embedding
        value_embedded = self.value_embedding(x)

        # Apply positional embedding if specified
        if self.positional_embedding is not None:
            positional_info = self.positional_embedding(value_embedded)
            combined_embedding = value_embedded + positional_info
        else:
            combined_embedding = value_embedded

        return combined_embedding

    def get_config(self) -> Dict[str, Any]:
        config = super(DataEmbedding, self).get_config()
        config.update(
            {
                "embed_size": self.embed_size,
                "positional_type": self.positional_type,
            }
        )
        return config

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[0], input_shape[1], self.embed_size


class TokenEmbedding(tf.keras.layers.Layer):
    """
    A layer that performs token embedding, equivalent to a dense layer for time series data.

    This layer transforms input features into an embedding space of specified dimension.
    It applies a linear transformation to the last dimension of the input tensor.

    Args:
        embed_size (int): The size of the embedding output dimension.

    Input shape:
        - 3D tensor with shape `(batch_size, time_steps, input_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, time_steps, embed_size)`

    Example:
        ```python
        # Create a TokenEmbedding layer with embedding size of 256
        embedding_layer = TokenEmbedding(embed_size=256)

        # Apply to input with shape (batch_size, sequence_length, features)
        input_tensor = tf.random.normal((32, 100, 10))
        output_tensor = embedding_layer(input_tensor)  # Shape: (32, 100, 256)
        ```
    """

    def __init__(self, embed_size: int, **kwargs):
        """
        Initialize the TokenEmbedding layer.

        Args:
            embed_size (int): The size of the embedding output dimension.
        """
        super(TokenEmbedding, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.token_weights = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's weights based on input shape.

        Args:
            input_shape (Tuple[Optional[int], ...]): The input shape to the layer.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("The last dimension of input_shape must be defined.")

        self.token_weights = self.add_weight(
            name="token_weights",
            shape=[input_shape[-1], self.embed_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=self.embed_size**-0.5),
            trainable=True,
            dtype=self.dtype,
        )
        super(TokenEmbedding, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Performs the token embedding transformation.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, time_steps, input_dim).

        Returns:
            tf.Tensor: Embedded tensor of shape (batch_size, time_steps, embed_size).
        """
        # Use einsum for efficient matrix multiplication
        # 'b' - batch, 's' - sequence/time steps, 'f' - features, 'k' - embedding dimension
        y = tf.einsum("bsf,fk->bsk", x, self.token_weights)
        return y

    def get_config(self) -> Dict[str, Any]:
        config = {"embed_size": self.embed_size}
        base_config = super(TokenEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[0], input_shape[1], self.embed_size


class TokenRnnEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_size: int) -> None:
        super().__init__()
        self.embed_size = embed_size

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.rnn = GRU(self.embed_size, return_sequences=True, return_state=True)
        super().build(input_shape)

    def call(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape `(batch_size, sequence_length, input_dim)`.

        Returns
        -------
        y : tf.Tensor
            Output tensor of shape `(batch_size, sequence_length, embed_size)`.
        """
        y, _ = self.rnn(x)
        return y

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(TokenRnnEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TemporalEmbedding(tf.keras.layers.Layer):
    # minute, hour, weekday, day, month
    def __init__(self, embed_size, embed_type="fixed") -> None:
        super().__init__()
        self.minute_size = 6  # every 10 minutes
        self.hour_size = 24  #

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super().build(input_shape)
        self.minute_embed = Embedding(self.minute_size, 3)
        self.hour_embed = Embedding(self.hour_size, 6)

    def call(self, x, **kwargs):
        """Temporal related embedding

        Parameters
        ----------
        x : tf.Tensor
            _description_
        """
        return


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
