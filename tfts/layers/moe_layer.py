"""Layer for :py:class:`~tfts.models.transformer`"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Dense


class MoELayer(tf.keras.layers.Layer):
    """Mixture of Experts layer for time series prediction.

    This layer implements a Mixture of Experts architecture where multiple expert networks
    specialize in different patterns of the time series, and a gating network determines
    which experts to use for each prediction.
    """

    def __init__(
        self,
        num_experts: int,
        expert_hidden_size: int,
        gating_hidden_size: int,
        expert_activation: str = "relu",
        gating_activation: str = "softmax",
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        use_bias: bool = True,
        bias_initializer: str = "zeros",
        trainable: bool = True,
        name: Optional[str] = None,
    ):
        super(MoELayer, self).__init__(trainable=trainable, name=name)
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.gating_hidden_size = gating_hidden_size
        self.expert_activation = expert_activation
        self.gating_activation = gating_activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

    def build(self, input_shape: Tuple[int, ...]):
        input_dim = int(input_shape[-1])

        # Create expert networks
        self.experts = []
        for i in range(self.num_experts):
            expert = Dense(
                self.expert_hidden_size,
                activation=self.expert_activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=regularizers.get(self.kernel_regularizer),
                kernel_constraint=constraints.get(self.kernel_constraint),
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                name=f"expert_{i}",
            )
            self.experts.append(expert)

        # Create gating network
        self.gating_network = Dense(
            self.num_experts,
            activation=self.gating_activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizers.get(self.kernel_regularizer),
            kernel_constraint=constraints.get(self.kernel_constraint),
            use_bias=self.use_bias,
            bias_initializer=self.bias_initializer,
            name="gating_network",
        )

        # Output projection layer
        self.output_projection = Dense(
            input_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizers.get(self.kernel_regularizer),
            kernel_constraint=constraints.get(self.kernel_constraint),
            use_bias=self.use_bias,
            bias_initializer=self.bias_initializer,
            name="output_projection",
        )

        super(MoELayer, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the MoE layer.

        Args:
            inputs: Tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            output: Tensor of shape (batch_size, sequence_length, input_dim)
        """

        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(inputs)  # (batch_size, seq_length, expert_hidden_size)
            expert_outputs.append(expert_output)

        # Stack expert outputs
        expert_outputs = tf.stack(expert_outputs, axis=2)  # (batch_size, seq_length, num_experts, expert_hidden_size)

        # Get gating weights
        gating_weights = self.gating_network(inputs)  # (batch_size, seq_length, num_experts)
        gating_weights = tf.expand_dims(gating_weights, axis=-1)  # (batch_size, seq_length, num_experts, 1)

        # Combine expert outputs using gating weights
        combined_output = tf.reduce_sum(
            expert_outputs * gating_weights, axis=2
        )  # (batch_size, seq_length, expert_hidden_size)

        # Project back to input dimension
        output = self.output_projection(combined_output)  # (batch_size, seq_length, input_dim)

        return output

    def get_config(self) -> Dict[str, Any]:
        config = {
            "num_experts": self.num_experts,
            "expert_hidden_size": self.expert_hidden_size,
            "gating_hidden_size": self.gating_hidden_size,
            "expert_activation": self.expert_activation,
            "gating_activation": self.gating_activation,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "use_bias": self.use_bias,
            "bias_initializer": self.bias_initializer,
        }
        base_config = super(MoELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
