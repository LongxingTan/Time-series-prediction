import unittest

import tensorflow as tf

from tfts.layers.position_layer import PositionalEmbedding, PositionalEncoding, RotaryPositionEmbedding


class PositionLayerTest(unittest.TestCase):
    def test_position_encoding(self):
        layer = PositionalEncoding(max_len=512)
        x = tf.random.normal([2, 128, 512])
        positional_encoding = layer(x)
        self.assertEqual(positional_encoding.shape, (128, 512))

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 6))
        # plt.matshow(positional_encoding[0], fignum=1)
        # plt.title('Positional Encoding Tensor')
        # plt.xlabel('Embedding Dimension')
        # plt.ylabel('Sequence Position')
        # plt.colorbar(label='Encoding Value')
        # plt.show()
