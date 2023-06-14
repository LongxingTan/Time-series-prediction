import unittest

from tfts.layers.mask_layer import CausalMask, ProbMask


class MaskLayerTest(unittest.TestCase):
    def test_casual_mask_layer(self):
        B = 2 * 8
        L = 99
        mask = CausalMask(B, L).mask
        self.assertEqual(mask.shape, (B, L, L))
