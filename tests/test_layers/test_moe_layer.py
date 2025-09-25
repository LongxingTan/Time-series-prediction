import unittest

import tensorflow as tf

from tfts.layers.moe_layer import MoELayer


class TestMoELayer(tf.test.TestCase):
    def test_moe_layer_output_shape(self):
        """
        Tests if the output shape of the MoELayer is the same as the input shape.
        """
        # Define layer parameters
        num_experts = 4
        expert_hidden_size = 16
        gating_hidden_size = 8

        # Define input shape
        batch_size = 2
        sequence_length = 10
        input_dim = 5

        inputs = tf.random.uniform((batch_size, sequence_length, input_dim))
        moe_layer = MoELayer(
            num_experts=num_experts,
            expert_hidden_size=expert_hidden_size,
            gating_hidden_size=gating_hidden_size,
        )
        output = moe_layer(inputs)
        self.assertEqual(output.shape, inputs.shape)
