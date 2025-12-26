from typing import List

import tensorflow as tf


class MultiQuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles: List[float], name="multi_quantile_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        """
        y_true: [batch, pred_len, num_labels]
        y_pred: [batch, pred_len, num_labels * num_quantiles]
        """
        # Reshape y_pred to [batch, pred_len, num_labels, num_quantiles]
        # and y_true to [batch, pred_len, num_labels, 1]
        y_true = tf.expand_dims(y_true, axis=-1)

        # Split y_pred into the different quantiles
        # Assuming the head outputs quantiles stacked in the last dimension
        num_labels = y_true.shape[-2]
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[1], num_labels, len(self.quantiles)])

        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true[..., 0] - y_pred[..., i]
            # Pinball loss: max(q*e, (q-1)*e)
            quantile_l = tf.maximum(q * error, (q - 1) * error)
            losses.append(tf.reduce_mean(quantile_l))

        return tf.add_n(losses)
