"""Layer for :py:class:`~tfts.models.transformer`"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """Positional embedding layer that adds positional information to input embeddings.

    This layer implements the sinusoidal positional encoding as described in the paper
    "Attention Is All You Need" (Vaswani et al., 2017). It adds positional information
    to the input embeddings using sine and cosine functions of different frequencies.

    Args:
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
        name (str, optional): Layer name. Defaults to None.

    Input shape:
        - 3D tensor with shape `(batch_size, sequence_length, embedding_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, sequence_length, embedding_dim)`
    """

    def __init__(self, max_len: int = 5000, name: Optional[str] = None):
        super(PositionalEmbedding, self).__init__(name=name)
        self.max_len = max_len
        self.position_enc = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by pre-computing the positional encodings.

        Args:
            input_shape: Shape of the input tensor
        """
        super(PositionalEmbedding, self).build(input_shape)
        E = input_shape[-1]  # embedding dimension

        # Pre-compute positional encodings
        position_enc = np.array(
            [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)]
        )
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        self.position_enc = tf.convert_to_tensor(position_enc, dtype=tf.float32)

    def call(self, x: tf.Tensor, masking: bool = True) -> tf.Tensor:
        """Applies positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            masking: If True, applies masking to the output tensor. Defaults to True.

        Returns:
            Output tensor of the same shape as the input tensor, after applying positional encoding.
        """
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]

        # Get position indices for each sequence
        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])

        # Lookup positional encodings
        outputs = tf.nn.embedding_lookup(self.position_enc, position_ind)

        # Apply masking if requested
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)

        return tf.cast(outputs, tf.float32)

    def get_config(self) -> Dict[str, Union[int, str]]:
        """Get layer configuration.

        Returns:
            Dictionary containing layer configuration.
        """
        config = {"max_len": self.max_len}
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer that adds positional information to input embeddings.

    This layer implements a more efficient version of positional encoding that computes
    the encodings on-the-fly using matrix operations. It's particularly useful for
    variable-length sequences as it doesn't require pre-computing encodings for all
    possible positions.

    Args:
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
        name (str, optional): Layer name. Defaults to None.

    Input shape:
        - 3D tensor with shape `(batch_size, sequence_length, embedding_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, sequence_length, embedding_dim)`
    """

    def __init__(self, max_len: int = 5000, name: Optional[str] = None):
        super(PositionalEncoding, self).__init__(name=name)
        self.max_len = max_len

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of the input tensor
        """
        super(PositionalEncoding, self).build(input_shape)

    def call(self, x: tf.Tensor, masking: bool = True) -> tf.Tensor:
        """Applies positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            masking: If True, applies masking to the output tensor. Defaults to True.

        Returns:
            Output tensor of the same shape as the input tensor, after applying positional encoding.
        """
        d_model = x.get_shape().as_list()[-1]  # embedding dimension
        depth = d_model // 2
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]

        with tf.name_scope("positional_encode"):
            # Create position indices
            positions = tf.range(seq_length, dtype=tf.float32)[..., tf.newaxis]  # (seq_length, 1)

            # Create depth indices
            depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth  # (1, depth)

            # Calculate angle rates
            angle_rates = 1 / tf.pow(10000.0, depths)  # (1, depth)

            # Calculate angle radians
            angle_rads = tf.matmul(positions, angle_rates)  # (seq_length, depth)

            # Create positional encodings
            position_enc = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)  # (seq_length, d_model)

            # Expand for batch dimension
            position_enc = tf.expand_dims(position_enc, 0)  # (1, seq_length, d_model)
            position_enc = tf.tile(position_enc, [batch_size, 1, 1])  # (batch_size, seq_length, d_model)

            # Apply masking if requested
            if masking:
                position_enc = tf.where(tf.equal(x, 0), x, position_enc)

            return position_enc

    def get_config(self) -> Dict[str, Union[int, str]]:
        """Get layer configuration.

        Returns:
            Dictionary containing layer configuration.
        """
        config = {"max_len": self.max_len}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape


class RelativePositionEmbedding(tf.keras.layers.Layer):
    """Relative position embedding layer that adds relative positional information.

    This layer implements relative position embeddings as described in the paper
    "Self-Attention with Relative Position Representations" (Shaw et al., 2018).
    It learns embeddings for relative positions between query and key positions.

    Args:
        max_len (int, optional): Maximum sequence length. Defaults to 512.
        output_dim (int, optional): Output embedding dimension. Defaults to 512.
        name (str, optional): Layer name. Defaults to None.

    Input shape:
        - Tuple of two tensors:
            - Query tensor of shape `(batch_size, query_length, embedding_dim)`
            - Value tensor of shape `(batch_size, value_length, embedding_dim)`

    Output shape:
        - Tensor of shape `(batch_size, query_length, value_length, output_dim)`
    """

    def __init__(self, max_len: int = 512, output_dim: int = 512, name: Optional[str] = None):
        super(RelativePositionEmbedding, self).__init__(name=name)
        self.max_len = max_len
        self.output_dim = output_dim
        self.input_dim = max_len * 2 - 1  # Total number of relative positions

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """Build the layer by creating the embedding weights.

        Args:
            input_shape: Shape of the input tensors
        """
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embedding_initializer = tf.keras.initializers.get("zeros")
        self.embeddings = self.add_weight(
            name="relative_position_embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embedding_initializer,
            trainable=True,
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Applies relative position embeddings to the input tensors.

        Args:
            inputs: Tuple of (query_tensor, value_tensor) where:
                - query_tensor: Shape (batch_size, query_length, embedding_dim)
                - value_tensor: Shape (batch_size, value_length, embedding_dim)

        Returns:
            Tensor of shape (batch_size, query_length, value_length, output_dim)
            containing relative position embeddings.
        """
        q, v = inputs
        q_length = tf.shape(q)[1]
        v_length = tf.shape(v)[1]

        # Create position indices
        q_idx = tf.range(q_length, dtype=tf.int32)[:, tf.newaxis]  # (q_length, 1)
        v_idx = tf.range(v_length, dtype=tf.int32)[tf.newaxis, :]  # (1, v_length)

        # Calculate relative positions
        position_idx = v_idx - q_idx  # (q_length, v_length)

        # Clip positions to valid range
        max_position = (self.input_dim - 1) // 2
        position_idx = tf.clip_by_value(position_idx, -max_position, max_position)

        # Shift to positive indices
        position_idx = position_idx + max_position

        # Lookup embeddings
        embeddings = tf.gather(self.embeddings, position_idx)  # (q_length, v_length, output_dim)

        # Add batch dimension
        batch_size = tf.shape(q)[0]
        embeddings = tf.expand_dims(embeddings, 0)  # (1, q_length, v_length, output_dim)
        embeddings = tf.tile(embeddings, [batch_size, 1, 1, 1])  # (batch_size, q_length, v_length, output_dim)

        return embeddings

    def get_config(self) -> Dict[str, Union[int, str]]:
        """Get layer configuration.

        Returns:
            Dictionary containing layer configuration.
        """
        config = {"max_len": self.max_len, "output_dim": self.output_dim}
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(
        self, input_shape: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    ) -> Tuple[int, int, int, int]:
        query_shape, value_shape = input_shape
        return (query_shape[0], query_shape[1], value_shape[1], self.output_dim)


class RotaryPositionEmbedding(tf.keras.layers.Layer):
    """Rotary position embedding layer that adds rotary positional information.

    This layer implements rotary position embeddings (RoPE) as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021).
    It applies a rotation to the input embeddings based on their positions.

    Args:
        dim (int): Dimension of the input embeddings.
        name (str, optional): Layer name. Defaults to None.

    Input shape:
        - 3D tensor with shape `(batch_size, sequence_length, embedding_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, sequence_length, embedding_dim)`
    """

    def __init__(self, dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = dim

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of the input tensor
        """
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, cache_key: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Applies rotary position embeddings to the input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            cache_key: Optional tensor for caching position information. Defaults to None.

        Returns:
            Output tensor of the same shape as the input tensor, after applying rotary position embeddings.
        """
        batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]

        # Create position indices
        positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]  # (seq_length, 1)

        # Create dimension indices
        dims = tf.range(self.dim // 2, dtype=tf.float32)[tf.newaxis, :]  # (1, dim/2)

        # Calculate angle rates
        angle_rates = 1 / tf.pow(10000.0, 2 * dims / self.dim)  # (1, dim/2)

        # Calculate angle radians
        angle_rads = tf.matmul(positions, angle_rates)  # (seq_length, dim/2)

        # Create rotation matrices
        cos = tf.cos(angle_rads)  # (seq_length, dim/2)
        sin = tf.sin(angle_rads)  # (seq_length, dim/2)

        # Reshape inputs for rotation
        x = tf.reshape(inputs, [batch_size, seq_length, -1, 2])  # (batch_size, seq_length, dim/2, 2)

        # Apply rotation
        x1, x2 = tf.unstack(x, axis=-1)  # (batch_size, seq_length, dim/2)
        rotated = tf.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)  # (batch_size, seq_length, dim/2, 2)

        # Reshape back to original shape
        outputs = tf.reshape(rotated, [batch_size, seq_length, self.dim])

        return outputs

    def get_config(self) -> Dict[str, Union[int, str]]:
        """Get layer configuration.

        Returns:
            Dictionary containing layer configuration.
        """
        config = {"dim": self.dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape
