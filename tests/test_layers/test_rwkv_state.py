"""Focused regression coverage for the model review."""

import unittest

import numpy as np
import tensorflow as tf

from tfts.models.rwkv import RWKV, RWKVConfig


class RWKVStateTest(unittest.TestCase):
    def test_rwkv_graph_chunk_equivalence(self):
        model = RWKV(1, RWKVConfig(hidden_size=8, num_layers=1))
        block = model.blocks[0]
        x = tf.random.normal([2, 7, 8])
        state = model.init_state(2)[0]
        whole, final_state = block(x, state)

        @tf.function
        def step(x, state):
            return block(x, state)

        pieces = []
        for index in range(7):
            output, state = step(x[:, index : index + 1], state)
            pieces.append(output)
        np.testing.assert_allclose(tf.concat(pieces, 1), whole, atol=2e-5)
        for actual, expected in zip(tf.nest.flatten(state), tf.nest.flatten(final_state)):
            np.testing.assert_allclose(actual, expected, atol=2e-5)
