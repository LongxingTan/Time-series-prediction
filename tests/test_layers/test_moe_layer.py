import unittest

import tensorflow as tf

from tfts.layers.moe_layer import SparseMoe


class TestMoELayer(tf.test.TestCase):
    def test_moe_layer_output_shape(self):
        """
        Tests if the output shape of the MoELayer is the same as the input shape.
        """
        # Define layer parameters
        hidden_size_val = 768
        num_experts_val = 8
        num_experts_per_tok_val = 2
        moe_intermediate_size_val = 3072
        shared_expert_intermediate_size_val = 3072
        norm_topk_prob_val = True
        hidden_act_val = "silu"

        moe_layer = SparseMoe(
            hidden_size=hidden_size_val,
            num_experts=num_experts_val,
            num_experts_per_tok=num_experts_per_tok_val,
            moe_intermediate_size=moe_intermediate_size_val,
            shared_expert_intermediate_size=shared_expert_intermediate_size_val,
            norm_topk_prob=norm_topk_prob_val,
            hidden_act=hidden_act_val,
        )

        input_tensor = tf.random.normal((1, 10, hidden_size_val), dtype=tf.float32)
        output, router_logits = moe_layer(input_tensor)

        print("Input shape:", input_tensor.shape)
        print("Output shape:", output.shape)
        print("Router logits shape:", router_logits.shape)
