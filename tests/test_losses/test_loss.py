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


if __name__ == "__main__":
    unittest.main()
