from typing import List, Optional, Union

import tensorflow as tf


class MultiQuantileLoss(tf.keras.losses.Loss):
    """Multi-quantile loss using the pinball loss function.

    Computes the pinball (quantile) loss for each specified quantile and
    sums across quantiles.  Handles multi-horizon, multi-target outputs.

    Args:
        quantiles: List of quantile fractions, e.g. ``[0.1, 0.5, 0.9]``.
        name: Loss name.

    Shape:
        - ``y_true``: ``(batch, pred_len, num_labels)``
        - ``y_pred``: ``(batch, pred_len, num_labels * len(quantiles))``
    """

    def __init__(self, quantiles: List[float], name: str = "multi_quantile_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.expand_dims(y_true, axis=-1)
        num_labels = y_true.shape[-2]
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[1], num_labels, len(self.quantiles)])

        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true[..., 0] - y_pred[..., i]
            # Pinball loss: max(q*e, (q-1)*e)
            quantile_l = tf.maximum(q * error, (q - 1) * error)
            losses.append(tf.reduce_mean(quantile_l))

        return tf.add_n(losses)


def smape_loss(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    """Elementwise symmetric MAPE loss (SMAPE), returned per-sample (no reduction).

    Returns ``(batch, pred_len, num_features)`` so Keras/`fit` can apply a
    per-horizon ``sample_weight`` (e.g. a validity mask). Matches the M4 SMAPE
    convention ``200 * |y_pred - y_true| / (|y_true| + |y_pred| + eps)``.
    """
    den = tf.abs(y_true) + tf.abs(y_pred) + eps
    return 200.0 * tf.abs(y_pred - y_true) / den
