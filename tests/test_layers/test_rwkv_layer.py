import unittest

import tensorflow as tf

from tfts.layers.rwkv_layer import ChannelMixing, TimeMixing
from tfts.models.rwkv import RWKVConfig


class TimeMixingTest(unittest.TestCase):
    """Test cases for TimeMixing layer."""

    def setUp(self):
        self.config = RWKVConfig(hidden_size=64)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = self.config.hidden_size

    def _get_initial_state(self):
        """Helper to create the correct 4-part state for TimeMixing."""
        return [
            tf.zeros([self.batch_size, self.hidden_size]),  # last_x
            tf.zeros([self.batch_size, self.hidden_size]),  # aa
            tf.zeros([self.batch_size, self.hidden_size]),  # bb
            tf.fill([self.batch_size, self.hidden_size], -1e30),  # pp
        ]

    def test_initialization(self):
        layer = TimeMixing(self.config)
        self.assertEqual(layer.n_embd, self.config.hidden_size)

    def test_build(self):
        layer = TimeMixing(self.config)
        x = tf.random.normal([self.batch_size, self.seq_len, self.hidden_size])
        state = self._get_initial_state()
        output, new_state = layer(x, state)

        self.assertTrue(layer.built)
        self.assertIsNotNone(layer.time_mix_k)

    def test_forward_pass(self):
        layer = TimeMixing(self.config)
        x = tf.random.normal([self.batch_size, self.seq_len, self.hidden_size])
        state = self._get_initial_state()
        output, new_state = layer(x, state)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(len(new_state), 4)
        for s in new_state:
            self.assertEqual(s.shape, (self.batch_size, self.hidden_size))

    def test_state_update(self):
        layer = TimeMixing(self.config)
        x = tf.random.normal([self.batch_size, self.seq_len, self.hidden_size])
        initial_state = self._get_initial_state()
        _, new_state = layer(x, initial_state)

        # Check that state changed (using index 1 'aa' or index 0 'last_x')
        self.assertFalse(tf.reduce_all(tf.equal(initial_state[0], new_state[0])))
        self.assertFalse(tf.reduce_all(tf.equal(initial_state[1], new_state[1])))

    def test_gradient_flow(self):
        layer = TimeMixing(self.config)
        x = tf.Variable(tf.random.normal([self.batch_size, self.seq_len, self.hidden_size]))
        state = self._get_initial_state()

        with tf.GradientTape() as tape:
            output, _ = layer(x, state)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, x)
        self.assertIsNotNone(gradients)


class ChannelMixingTest(unittest.TestCase):
    """Test cases for ChannelMixing layer."""

    def setUp(self):
        self.config = RWKVConfig(hidden_size=64)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = self.config.hidden_size

    def test_forward_pass(self):
        layer = ChannelMixing(self.config)
        x = tf.random.normal([self.batch_size, self.seq_len, self.hidden_size])
        state = tf.zeros([self.batch_size, self.hidden_size])

        output, new_state = layer(x, state)

        # Output is 3D
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        # State is 2D (the last vector of the sequence)
        self.assertEqual(new_state.shape, (self.batch_size, self.hidden_size))

    def test_state_output(self):
        layer = ChannelMixing(self.config)
        x = tf.random.normal([self.batch_size, self.seq_len, self.hidden_size])
        state = tf.zeros([self.batch_size, self.hidden_size])

        _, new_state = layer(x, state)

        # In ChannelMixing, new_state should be the last timestep of input x
        tf.debugging.assert_near(new_state, x[:, -1, :])

    def test_gradient_flow(self):
        layer = ChannelMixing(self.config)
        x = tf.Variable(tf.random.normal([self.batch_size, self.seq_len, self.hidden_size]))
        state = tf.zeros([self.batch_size, self.hidden_size])

        with tf.GradientTape() as tape:
            output, _ = layer(x, state)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, x)
        self.assertIsNotNone(gradients)


if __name__ == "__main__":
    unittest.main()
