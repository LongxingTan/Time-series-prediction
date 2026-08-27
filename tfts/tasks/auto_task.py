"""Small, reusable task heads."""

from typing import Optional, Sequence, Tuple

import tensorflow as tf

from tfts.distributions import DistributionOutput

from .base import BaseHead


class PointForecastHead(BaseHead):
    def __init__(self, prediction_length: int, target_dim: int = 1, residual: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = int(prediction_length)
        self.target_dim = int(target_dim)
        self.residual = residual
        self.projection = tf.keras.layers.Dense(self.target_dim)

    def call(self, sequence_output, past_values=None, **kwargs):
        states = sequence_output[:, -self.prediction_length :, :]
        predictions = self.projection(states)
        if self.residual is None:
            return predictions
        if past_values is None:
            raise ValueError("past_values is required for residual forecasting")
        target = past_values[..., : self.target_dim]
        if self.residual == "last_value":
            baseline = tf.tile(target[:, -1:, :], [1, self.prediction_length, 1])
        elif self.residual == "mean":
            baseline = tf.tile(tf.reduce_mean(target, axis=1, keepdims=True), [1, self.prediction_length, 1])
        elif self.residual == "last_window":
            baseline = target[:, -self.prediction_length :, :]
        else:
            raise ValueError("Unknown residual %r" % self.residual)
        return predictions + tf.cast(baseline, predictions.dtype)


class QuantileForecastHead(BaseHead):
    def __init__(self, prediction_length: int, quantiles: Sequence[float], target_dim: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = int(prediction_length)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.target_dim = int(target_dim)
        self.projection = tf.keras.layers.Dense(self.target_dim * len(self.quantiles))

    def call(self, sequence_output, **kwargs):
        states = sequence_output[:, -self.prediction_length :, :]
        values = self.projection(states)
        shape = tf.concat([tf.shape(values)[:-1], [self.target_dim, len(self.quantiles)]], axis=0)
        return tf.reshape(values, shape)


class DistributionForecastHead(BaseHead):
    def __init__(self, distribution: DistributionOutput, prediction_length: int, **kwargs):
        super().__init__(**kwargs)
        self.distribution = distribution
        self.prediction_length = int(prediction_length)

    def call(self, sequence_output, **kwargs):
        return self.distribution.parameters(sequence_output[:, -self.prediction_length :, :])


class ClassificationHead(BaseHead):
    """Masked pooling followed by logits; probabilities stay in postprocessing."""

    def __init__(self, num_labels: int, hidden_units: Tuple[int, ...] = (128,), dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = [tf.keras.layers.Dense(units, activation="gelu") for units in hidden_units]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(num_labels)

    def call(self, sequence_output, padding_mask=None, training=None, **kwargs):
        if padding_mask is None:
            pooled = tf.reduce_mean(sequence_output, axis=1)
        else:
            mask = tf.cast(padding_mask, sequence_output.dtype)[..., None]
            pooled = tf.math.divide_no_nan(tf.reduce_sum(sequence_output * mask, axis=1), tf.reduce_sum(mask, axis=1))
        for layer in self.hidden_layers:
            pooled = layer(pooled)
        return self.classifier(self.dropout(pooled, training=training))


class ReconstructionHead(BaseHead):
    def __init__(self, target_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.projection = tf.keras.layers.Dense(target_dim)

    def call(self, sequence_output, **kwargs):
        return self.projection(sequence_output)


# Concise aliases for callers that prefer the older noun order.
PredictionHead = PointForecastHead
QuantileHead = QuantileForecastHead
DistributionHead = DistributionForecastHead
