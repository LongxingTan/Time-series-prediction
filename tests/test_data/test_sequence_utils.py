import unittest

import numpy as np

from tfts.data import generate_sequence_windows, pad_sequence, pad_sequences, sequence_mask


class SequenceUtilsTest(unittest.TestCase):
    def test_sequence_mask_uses_true_for_valid_steps(self):
        np.testing.assert_array_equal(
            sequence_mask([2, 4], max_length=4),
            [[True, True, False, False], [True, True, True, True]],
        )

    def test_pad_sequence_left_keeps_latest_context_and_returns_mask(self):
        values, mask = pad_sequence([1.0, 2.0, 3.0], 5, padding_side="left", return_padding_mask=True)
        np.testing.assert_array_equal(values[:, 0], [0.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(mask, [False, False, True, True, True])

        values, mask = pad_sequence([1.0, 2.0, 3.0, 4.0], 2, padding_side="left", return_padding_mask=True)
        np.testing.assert_array_equal(values[:, 0], [3.0, 4.0])
        np.testing.assert_array_equal(mask, [True, True])

    def test_pad_sequences_supports_multivariate_inputs(self):
        values, mask = pad_sequences(
            [np.array([[1.0, 10.0], [2.0, 20.0]]), np.array([[3.0, 30.0]])],
            padding_side="right",
            return_padding_mask=True,
        )
        self.assertEqual(values.shape, (2, 2, 2))
        np.testing.assert_array_equal(values[1, 1], [0.0, 0.0])
        np.testing.assert_array_equal(mask, [[True, True], [True, False]])

    def test_generate_sequence_windows_skips_irregular_boundaries(self):
        windows = generate_sequence_windows(
            length=5,
            context_length=2,
            prediction_length=1,
            time_values=[0, 1, 2, 4, 5],
        )
        self.assertEqual(len(windows), 1)
        np.testing.assert_array_equal(windows[0][0], [0, 1])
        np.testing.assert_array_equal(windows[0][1], [2])

    def test_generate_sequence_windows_inference_uses_empty_decoder(self):
        windows = generate_sequence_windows(5, 3, 2, mode="inference")
        self.assertEqual(len(windows), 3)
        np.testing.assert_array_equal(windows[-1][0], [2, 3, 4])
        self.assertEqual(windows[-1][1].shape, (0,))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            sequence_mask([-1])
        with self.assertRaises(ValueError):
            pad_sequence([1, 2], 2, padding_side="middle")
        with self.assertRaises(ValueError):
            pad_sequences([np.ones((2, 1)), np.ones((2, 2))])
        with self.assertRaises(ValueError):
            generate_sequence_windows(4, 2, 1, time_values=[0, 1])


if __name__ == "__main__":
    unittest.main()
