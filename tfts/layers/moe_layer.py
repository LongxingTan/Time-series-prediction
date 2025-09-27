"""Layer for :py:class:`~tfts.models.transformer`"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Dense

from tfts.layers.dense_layer import MoeMLP


class SparseMoe(tf.keras.layers.Layer):
    """Mixture of Experts layer for time series prediction.

    This layer implements a Mixture of Experts architecture where multiple expert networks
    specialize in different patterns of the time series, and a gating network determines
    which experts to use for each prediction.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self, input_shape: Tuple[int, ...]):
        # Gating network (router)
        self.gate = Dense(
            self.num_experts,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="gate",
        )

        # Experts
        self.experts = [
            MoeMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                hidden_act=self.hidden_act,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=self.use_bias,
                name=f"expert_{i}",
            )
            for i in range(self.num_experts)
        ]

        # Shared Expert
        self.shared_expert = MoeMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_expert_intermediate_size,
            hidden_act=self.hidden_act,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            use_bias=self.use_bias,
            name="shared_expert",
        )
        self.shared_expert_gate = Dense(
            1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="shared_expert_gate",
        )
        super().build(input_shape)

    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass of the MoE layer.

        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns:
            output: Tuple[Tensor (batch_size, sequence_length, hidden_dim), Tensor (num_tokens, num_experts)]
        """
        batch_size = tf.shape(hidden_states)[0]
        sequence_length = tf.shape(hidden_states)[1]
        hidden_dim = tf.shape(hidden_states)[2]

        hidden_states_flat = tf.reshape(hidden_states, (-1, hidden_dim))  # (num_tokens, hidden_dim)

        router_logits = self.gate(hidden_states_flat)  # (num_tokens, num_experts)

        routing_weights = tf.nn.softmax(router_logits, axis=-1)

        # Get top-k experts and their weights
        routing_weights, selected_experts = tf.math.top_k(routing_weights, k=self.top_k)  # (num_tokens, top_k)

        if self.norm_topk_prob:
            # Normalize top-k probabilities
            routing_weights = routing_weights / tf.reduce_sum(routing_weights, axis=-1, keepdims=True)

        routing_weights = tf.cast(routing_weights, hidden_states.dtype)

        final_hidden_states = tf.zeros_like(hidden_states_flat)

        for expert_idx in tf.range(self.num_experts):
            # Create a boolean mask for tokens that selected the current expert
            expert_chosen_mask = tf.equal(selected_experts, expert_idx)  # (num_tokens, top_k)

            # --- DEBUGGING PRINT STATEMENTS ---
            # tf.print("Iteration for expert_idx:", expert_idx)
            # tf.print("Shape of selected_experts:", tf.shape(selected_experts))
            # tf.print("Shape of expert_chosen_mask:", tf.shape(expert_chosen_mask))
            # tf.print("Rank of expert_chosen_mask:", tf.rank(expert_chosen_mask))
            # --- END DEBUGGING PRINT STATEMENTS ---

            coordinates = tf.where(expert_chosen_mask)

            # Check if there are any true values for this expert
            if tf.shape(coordinates)[0] == 0:
                continue

            # Unpack the coordinates
            # coordinates[:, 0] gives the row indices (token indices)
            # coordinates[:, 1] gives the column indices (top-k positions)
            token_indices_for_expert = coordinates[:, 0]
            topk_position_for_expert = coordinates[:, 1]

            # Gather the hidden states for the tokens that chose this expert
            current_state = tf.gather(hidden_states_flat, token_indices_for_expert)
            current_hidden_states_expert_output = self.experts[expert_idx](current_state)

            # Gather the corresponding routing weights for these tokens and their chosen expert
            current_routing_weights = tf.gather_nd(
                routing_weights, tf.stack([token_indices_for_expert, topk_position_for_expert], axis=-1)
            )
            current_routing_weights = tf.expand_dims(
                current_routing_weights, axis=-1
            )  # Shape (num_tokens_for_expert, 1)

            weighted_expert_output = current_hidden_states_expert_output * tf.cast(
                current_routing_weights, current_hidden_states_expert_output.dtype
            )

            # Accumulate results using tf.tensor_scatter_nd_add
            indices_to_scatter = tf.expand_dims(token_indices_for_expert, axis=-1)  # Shape (num_tokens_for_expert, 1)
            final_hidden_states = tf.tensor_scatter_nd_add(
                final_hidden_states, indices_to_scatter, weighted_expert_output
            )

        # Shared Expert Computation
        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_expert_gate_output = tf.nn.sigmoid(self.shared_expert_gate(hidden_states_flat))
        shared_expert_output = shared_expert_gate_output * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = tf.reshape(final_hidden_states, (batch_size, sequence_length, hidden_dim))
        return final_hidden_states, router_logits

    def get_config(self) -> Dict[str, Any]:
        config = {
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.top_k,  # Using top_k for consistency with internal
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "norm_topk_prob": self.norm_topk_prob,
            "hidden_act": self.hidden_act,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "use_bias": self.use_bias,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
