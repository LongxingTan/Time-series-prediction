"""Tests for loss functions."""

import unittest

import numpy as np
import tensorflow as tf

from tfts.losses.loss import MultiQuantileLoss


class MultiQuantileLossTest(unittest.TestCase):
    """Test cases for MultiQuantileLoss."""

    def test_initialization(self):
        """Test loss initialization."""
        quantiles = [0.1, 0.5, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)
        self.assertEqual(loss.quantiles, quantiles)
        self.assertEqual(loss.name, "multi_quantile_loss")

    def test_custom_name(self):
        """Test loss initialization with custom name."""
        quantiles = [0.5]
        custom_name = "custom_quantile_loss"
        loss = MultiQuantileLoss(quantiles=quantiles, name=custom_name)
        self.assertEqual(loss.name, custom_name)

    def test_loss_shape(self):
        """Test that loss returns a scalar."""
        quantiles = [0.1, 0.5, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)

        # Create test data
        batch_size = 4
        pred_len = 10
        num_labels = 1
        num_quantiles = len(quantiles)

        y_true = tf.random.normal([batch_size, pred_len, num_labels])
        y_pred = tf.random.normal([batch_size, pred_len, num_labels * num_quantiles])

        # Compute loss
        loss_value = loss(y_true, y_pred)

        # Check that loss is a scalar
        self.assertEqual(loss_value.shape, ())
        self.assertTrue(tf.is_tensor(loss_value))

    def test_perfect_prediction(self):
        """Test that perfect predictions give low loss."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        # Create perfect prediction (median)
        y_true = tf.constant([[[1.0], [2.0], [3.0]]])
        y_pred = tf.constant([[[1.0], [2.0], [3.0]]])

        loss_value = loss(y_true, y_pred)
        self.assertLess(loss_value.numpy(), 0.01)  # Should be very close to 0

    def test_perfect_prediction_multiple_quantiles(self):
        """Test perfect predictions with multiple quantiles."""
        quantiles = [0.1, 0.5, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[1.0], [2.0], [3.0]]])
        # Perfect predictions for all quantiles
        y_pred = tf.constant([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]])

        loss_value = loss(y_true, y_pred)
        self.assertLess(loss_value.numpy(), 0.01)

    def test_multiple_quantiles(self):
        """Test with multiple quantiles."""
        quantiles = [0.1, 0.5, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)

        batch_size = 2
        pred_len = 5
        num_labels = 1
        num_quantiles = len(quantiles)

        y_true = tf.random.normal([batch_size, pred_len, num_labels])
        y_pred = tf.random.normal([batch_size, pred_len, num_labels * num_quantiles])

        loss_value = loss(y_true, y_pred)

        # Loss should be positive
        self.assertGreater(loss_value.numpy(), 0)

    def test_quantile_properties(self):
        """Test that quantile loss has correct properties."""
        quantiles = [0.9]  # High quantile
        loss = MultiQuantileLoss(quantiles=quantiles)

        # For q=0.9, overestimation should be penalized less than underestimation
        y_true = tf.constant([[[5.0]]])

        # Overestimate
        y_pred_over = tf.constant([[[6.0]]])
        loss_over = loss(y_true, y_pred_over)

        # Underestimate
        y_pred_under = tf.constant([[[4.0]]])
        loss_under = loss(y_true, y_pred_under)

        # For q=0.9, underestimation should be penalized more heavily
        self.assertGreater(loss_under.numpy(), loss_over.numpy())

    def test_low_quantile_properties(self):
        """Test that low quantile penalizes overestimation more."""
        quantiles = [0.1]  # Low quantile
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[5.0]]])

        # Overestimate
        y_pred_over = tf.constant([[[6.0]]])
        loss_over = loss(y_true, y_pred_over)

        # Underestimate
        y_pred_under = tf.constant([[[4.0]]])
        loss_under = loss(y_true, y_pred_under)

        # For q=0.1, overestimation should be penalized more heavily
        self.assertGreater(loss_over.numpy(), loss_under.numpy())

    def test_multiple_labels(self):
        """Test with multiple target labels."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        batch_size = 2
        pred_len = 5
        num_labels = 3
        num_quantiles = len(quantiles)

        y_true = tf.random.normal([batch_size, pred_len, num_labels])
        y_pred = tf.random.normal([batch_size, pred_len, num_labels * num_quantiles])

        loss_value = loss(y_true, y_pred)

        # Loss should be computed correctly
        self.assertGreater(loss_value.numpy(), 0)
        self.assertEqual(loss_value.shape, ())

    def test_batch_consistency(self):
        """Test that loss is consistent across batch sizes."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        # Same prediction repeated
        y_true_single = tf.constant([[[1.0], [2.0], [3.0]]])
        y_pred_single = tf.constant([[[1.5], [2.5], [3.5]]])

        # Batch of 2 with same data
        y_true_batch = tf.concat([y_true_single, y_true_single], axis=0)
        y_pred_batch = tf.concat([y_pred_single, y_pred_single], axis=0)

        loss_single = loss(y_true_single, y_pred_single)
        loss_batch = loss(y_true_batch, y_pred_batch)

        # Losses should be the same (mean over batch)
        np.testing.assert_allclose(loss_single.numpy(), loss_batch.numpy(), rtol=1e-5)

    def test_symmetric_quantile(self):
        """Test that median quantile (0.5) treats over/under prediction equally."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[5.0]]])

        # Overestimate by 1
        y_pred_over = tf.constant([[[6.0]]])
        loss_over = loss(y_true, y_pred_over)

        # Underestimate by 1
        y_pred_under = tf.constant([[[4.0]]])
        loss_under = loss(y_true, y_pred_under)

        # For median (q=0.5), should be symmetric
        np.testing.assert_allclose(loss_over.numpy(), loss_under.numpy(), rtol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        quantiles = [0.1, 0.5, 0.9]
        loss_fn = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[1.0], [2.0], [3.0]]])
        y_pred = tf.Variable([[[1.5, 2.0, 2.5], [2.5, 3.0, 3.5], [3.5, 4.0, 4.5]]])

        with tf.GradientTape() as tape:
            loss_value = loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss_value, y_pred)

        # Gradients should exist and not be None
        self.assertIsNotNone(gradients)
        self.assertFalse(tf.reduce_all(tf.equal(gradients, 0)))

    def test_single_quantile(self):
        """Test with single quantile."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[1.0], [2.0]]])
        y_pred = tf.constant([[[1.5], [2.5]]])

        loss_value = loss(y_true, y_pred)
        self.assertGreater(loss_value.numpy(), 0)

    def test_edge_quantiles(self):
        """Test with extreme quantiles (close to 0 and 1)."""
        quantiles = [0.01, 0.99]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[5.0]]])
        y_pred = tf.constant([[[4.0, 6.0]]])

        loss_value = loss(y_true, y_pred)
        self.assertGreater(loss_value.numpy(), 0)

    def test_zero_error(self):
        """Test loss when all predictions are perfect."""
        quantiles = [0.1, 0.5, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.zeros([2, 3, 1])
        y_pred = tf.zeros([2, 3, 3])

        loss_value = loss(y_true, y_pred)
        np.testing.assert_allclose(loss_value.numpy(), 0.0, atol=1e-7)

    def test_negative_values(self):
        """Test with negative values in predictions and targets."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[-5.0], [-2.0], [3.0]]])
        y_pred = tf.constant([[[-4.0], [-3.0], [2.0]]])

        loss_value = loss(y_true, y_pred)
        self.assertGreater(loss_value.numpy(), 0)

    def test_large_batch_size(self):
        """Test with large batch size."""
        quantiles = [0.1, 0.5, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)

        batch_size = 100
        pred_len = 20
        num_labels = 2

        y_true = tf.random.normal([batch_size, pred_len, num_labels])
        y_pred = tf.random.normal([batch_size, pred_len, num_labels * len(quantiles)])

        loss_value = loss(y_true, y_pred)
        self.assertEqual(loss_value.shape, ())
        self.assertGreater(loss_value.numpy(), 0)

    def test_many_quantiles(self):
        """Test with many quantiles."""
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[5.0]]])
        y_pred = tf.constant([[[4.5, 4.7, 4.9, 5.0, 5.0, 5.1, 5.3, 5.5, 5.7]]])

        loss_value = loss(y_true, y_pred)
        self.assertGreater(loss_value.numpy(), 0)

    def test_loss_magnitude_scales_with_error(self):
        """Test that larger errors produce larger losses."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[5.0]]])

        # Small error
        y_pred_small = tf.constant([[[5.5]]])
        loss_small = loss(y_true, y_pred_small)

        # Large error
        y_pred_large = tf.constant([[[10.0]]])
        loss_large = loss(y_true, y_pred_large)

        self.assertGreater(loss_large.numpy(), loss_small.numpy())

    def test_reshape_correctness(self):
        """Test that reshaping logic works correctly with different configurations."""
        quantiles = [0.25, 0.5, 0.75]
        loss = MultiQuantileLoss(quantiles=quantiles)

        # 2 labels, 3 quantiles
        y_true = tf.constant([[[1.0, 2.0], [3.0, 4.0]]])
        y_pred = tf.constant([[[1.1, 1.0, 0.9, 2.1, 2.0, 1.9], [3.1, 3.0, 2.9, 4.1, 4.0, 3.9]]])

        loss_value = loss(y_true, y_pred)
        self.assertGreater(loss_value.numpy(), 0)
        self.assertEqual(loss_value.shape, ())

    def test_loss_is_differentiable(self):
        """Test that loss produces finite gradients."""
        quantiles = [0.5]
        loss_fn = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[1.0], [2.0]]])
        y_pred = tf.Variable([[[1.5], [2.5]]])

        with tf.GradientTape() as tape:
            loss_value = loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss_value, y_pred)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(gradients)))

    def test_empty_batch_dimension(self):
        """Test behavior with batch size of 1."""
        quantiles = [0.5]
        loss = MultiQuantileLoss(quantiles=quantiles)

        y_true = tf.constant([[[5.0]]])
        y_pred = tf.constant([[[5.5]]])

        loss_value = loss(y_true, y_pred)
        self.assertGreater(loss_value.numpy(), 0)

    def test_quantile_ordering(self):
        """Test that quantile order doesn't affect total loss significantly."""
        quantiles_ordered = [0.1, 0.5, 0.9]
        quantiles_unordered = [0.5, 0.1, 0.9]

        loss_ordered = MultiQuantileLoss(quantiles=quantiles_ordered)
        loss_unordered = MultiQuantileLoss(quantiles=quantiles_unordered)

        y_true = tf.constant([[[5.0]]])
        # Predictions must match the quantile order
        y_pred_ordered = tf.constant([[[4.0, 5.0, 6.0]]])
        y_pred_unordered = tf.constant([[[5.0, 4.0, 6.0]]])

        loss_val_ordered = loss_ordered(y_true, y_pred_ordered)
        loss_val_unordered = loss_unordered(y_true, y_pred_unordered)

        # Total loss should be the same regardless of order
        np.testing.assert_allclose(loss_val_ordered.numpy(), loss_val_unordered.numpy(), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
