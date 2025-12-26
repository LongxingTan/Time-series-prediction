"""Layer for Graph Neural Networks"""

from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Dropout, Layer


class GraphConv(Layer):
    """Basic Graph Convolution Layer.

    This layer implements the graph convolution operation:
    Output = Activation(Adjacency * Features * Kernel + Bias)

    Parameters
    ----------
    units : int
        Dimensionality of the output space.
    activation : str or callable, optional
        Activation function to use. If you don't specify anything, no activation is applied.
    use_bias : bool, optional
        Whether the layer uses a bias vector. Defaults to True.
    kernel_initializer : str, optional
        Initializer for the `kernel` weights matrix. Defaults to "glorot_uniform".
    bias_initializer : str, optional
        Initializer for the `bias` vector. Defaults to "zeros".
    kernel_regularizer : str, optional
        Regularizer function applied to the `kernel` weights matrix.
    bias_regularizer : str, optional
        Regularizer function applied to the `bias` vector.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # input_shape[0] is features: (batch, nodes, features)
        # input_shape[1] is adjacency: (batch, nodes, nodes)
        feat_shape = input_shape[0]
        input_dim = feat_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        super(GraphConv, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            A tuple containing:
            - features: 3D tensor (batch_size, num_nodes, input_dim)
            - adjacency: 3D tensor (batch_size, num_nodes, num_nodes) or SparseTensor

        Returns
        -------
        tf.Tensor
            Output tensor (batch_size, num_nodes, units)
        """
        features, adjacency = inputs

        # Transform features: H = XW
        output = tf.matmul(features, self.kernel)

        # Propagate: O = AH
        # Handle Sparse Adjacency
        if isinstance(adjacency, tf.sparse.SparseTensor):
            # Sparse matmul requires 2D, typically used in single-graph mode or carefully reshaped batches
            # Assuming batch_size=1 or shared adjacency for simplicity in sparse mode,
            # otherwise standard dense matmul is safer for batched data.
            output = tf.sparse.sparse_dense_matmul(adjacency, output)
        else:
            output = tf.matmul(adjacency, output)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
        features_shape = input_shape[0]
        return features_shape[:-1] + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphAttention(Layer):
    """Graph Attention Layer (GAT).

    This layer implements the multi-head graph attention mechanism.
    Unlike the original code, this implementation is fully vectorized, computing
    all heads in parallel for better performance.

    Parameters
    ----------
    units : int
        Dimensionality of the output space per head (if reduction='concat') or total (if 'average').
    num_heads : int, optional
        Number of attention heads. Defaults to 1.
    head_reduction : str, optional
        How to combine heads: 'concat' or 'average'. Defaults to 'average'.
    dropout_rate : float, optional
        Dropout rate for attention coefficients. Defaults to 0.5.
    activation : str, optional
        Activation function. Defaults to "relu".
    use_bias : bool, optional
        Whether to use bias. Defaults to True.
    """

    def __init__(
        self,
        units: int,
        num_heads: int = 1,
        head_reduction: str = "average",
        dropout_rate: float = 0.5,
        activation: str = "relu",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(GraphAttention, self).__init__(**kwargs)
        if head_reduction not in {"concat", "average"}:
            raise ValueError("Possible reduction methods: concat, average")

        self.units = units
        self.num_heads = num_heads
        self.head_reduction = head_reduction
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Output dimension calculation
        if head_reduction == "concat":
            self.output_dim = self.units * self.num_heads
        else:
            self.output_dim = self.units

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # input_shape[0]: (Batch, Nodes, Features)
        input_dim = input_shape[0][-1]

        # W: Transformation kernel for all heads (Input -> Heads * Units)
        self.kernel = self.add_weight(
            shape=(input_dim, self.num_heads * self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name="kernel",
        )

        # W2: Residual kernel (Input -> Heads * Units) - matched from original logic
        self.kernel_residual = self.add_weight(
            shape=(input_dim, self.num_heads * self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name="kernel_residual",
        )

        # Attention Kernels
        # Self attention mechanism parameters (Heads, Units, 1)
        self.attn_kernel_self = self.add_weight(
            shape=(self.num_heads, self.units, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name="attn_kernel_self",
        )

        # Neighbor attention mechanism parameters (Heads, Units, 1)
        self.attn_kernel_neighs = self.add_weight(
            shape=(self.num_heads, self.units, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name="attn_kernel_neigh",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.num_heads * self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )

        self.dropout = Dropout(self.dropout_rate)
        super(GraphAttention, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of Graph Attention.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            - features: (batch_size, num_nodes, input_dim)
            - adjacency: (batch_size, num_nodes, num_nodes)
        training : bool, optional
            Whether in training mode (for dropout), by default None

        Returns
        -------
        tf.Tensor
            Output tensor
        """
        X, A = inputs
        # X shape: (B, N, F)
        # A shape: (B, N, N)

        # 1. Linear Transformations
        # (B, N, Heads * Units)
        features = tf.matmul(X, self.kernel)
        features_residual = tf.matmul(X, self.kernel_residual)

        # Reshape to (B, N, Heads, Units) to separate heads
        B = tf.shape(features)[0]
        N = tf.shape(features)[1]

        features_reshaped = tf.reshape(features, (B, N, self.num_heads, self.units))

        # 2. Attention Scores
        # Compute "a^T * Wh_i" and "a^T * Wh_j"
        # (B, N, Heads, Units) x (Heads, Units, 1) -> (B, N, Heads, 1)
        # We use einsum for clarity with batch and head dims
        attn_for_self = tf.einsum("bnhu,huo->bnh", features_reshaped, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("bnhu,huo->bnh", features_reshaped, self.attn_kernel_neighs)

        # Add scores (broadcasting): (B, N, 1, Heads) + (B, 1, N, Heads) -> (B, N, N, Heads)
        # Note: Original code logic sum dense matrices.
        dense = tf.expand_dims(attn_for_self, axis=2) + tf.expand_dims(attn_for_neighs, axis=1)

        # LeakyReLU
        dense = tf.nn.leaky_relu(dense, alpha=0.2)

        # 3. Masking and Softmax
        # Mask: -10e9 * (1.0 - A). Expand A to match heads: (B, N, N, 1)
        A_expanded = tf.expand_dims(A, axis=-1)
        mask = -10e9 * (1.0 - A_expanded)

        # Add mask to logits
        dense += mask

        # Softmax over neighbors (axis 2) -> (B, N, N, Heads)
        attn_coef = tf.nn.softmax(dense, axis=2)

        # Apply dropout to coefficients
        attn_coef = self.dropout(attn_coef, training=training)

        # Apply dropout to features (Original code applied dropout to features before aggregation)
        features_dropout = self.dropout(features_reshaped, training=training)

        # 4. Aggregation
        # (B, N, N, Heads) x (B, N, Heads, Units) -> (B, N, Heads, Units)
        # This represents: output_i = sum_j(alpha_ij * h_j)
        node_features = tf.einsum("bnkh,bkhu->bnhu", attn_coef, features_dropout)

        # Flatten heads back to (B, N, Heads * Units)
        node_features = tf.reshape(node_features, (B, N, self.num_heads * self.units))

        # 5. Residual Connection + Bias
        node_features += features_residual

        if self.use_bias:
            node_features = tf.nn.bias_add(node_features, self.bias)

        # 6. Reduce Heads
        if self.head_reduction == "concat":
            # Already in shape (B, N, Heads * Units)
            output = node_features
        else:
            # Average: Reshape to (B, N, Heads, Units) then mean
            output = tf.reshape(node_features, (B, N, self.num_heads, self.units))
            output = tf.reduce_mean(output, axis=2)

        # 7. Activation
        output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
        features_shape = input_shape[0]
        return features_shape[:-1] + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "units": self.units,
            "num_heads": self.num_heads,
            "head_reduction": self.head_reduction,
            "dropout_rate": self.dropout_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
