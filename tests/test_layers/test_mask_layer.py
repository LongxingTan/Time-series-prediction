import unittest

import tensorflow as tf

from tfts.layers.mask_layer import CausalMask, ProbMask


class MaskLayerTest(unittest.TestCase):
    def test_casual_mask_layer(self):
        B, L = 2, 4
        dummy = tf.zeros((B, L, 1))
        mask_layer = CausalMask(num_attention_heads=1)
        mask = mask_layer(dummy)
        self.assertEqual(mask.shape, (B, L, L))
