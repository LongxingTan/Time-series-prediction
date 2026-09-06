import unittest

import numpy as np

from tfts.data import final_windows, sampled_windows
from tfts.training import window_trainer


class WindowSamplingTest(unittest.TestCase):
    def test_final_context_is_left_padded(self):
        values, mask = final_windows([np.array([4.0, 5.0])], seq_len=4)
        np.testing.assert_array_equal(values[0, :, 0], [0, 0, 4, 5])
        np.testing.assert_array_equal(mask[0, :, 0], [0, 0, 1, 1])

    def test_random_windows_preserve_cutoff_and_target_padding(self):
        histories = [np.arange(1, 5, dtype=np.float32)]
        first = sampled_windows(histories, np.random.default_rng(7), seq_len=5, pred_len=5)
        second = sampled_windows(histories, np.random.default_rng(7), seq_len=5, pred_len=5)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        x, y, mask = first
        self.assertEqual(y[0, 0, 0], x[0, -1, 0] + 1)
        valid = int(mask.sum())
        np.testing.assert_array_equal(y[0, :valid, 0], np.arange(x[0, -1, 0] + 1, 5))
        np.testing.assert_array_equal(y[0, valid:, 0], 0)

    def test_trainer_imports_reuse_data_implementation(self):
        self.assertIs(window_trainer.final_windows, final_windows)
        self.assertIs(window_trainer.sampled_windows, sampled_windows)
