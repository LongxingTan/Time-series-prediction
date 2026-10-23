"""Small, reusable task heads."""

from typing import Optional, Sequence, Tuple

import tensorflow as tf

from tfts.contracts import ForecastOutput, OutputPort
from tfts.distributions import DistributionOutput, NormalOutput
from tfts.registry import ComponentSpec, register_head

from .base import BaseHead


class ForecastHead(BaseHead):
    """A registered forecast projection owns its output and default objective."""

    required_output_port = OutputPort.SEQUENCE
    default_objective = "mse"

    @classmethod
    def from_task_config(cls, config, **kwargs):
        return cls(config.output_chunk_length, target_dim=config.target_dim, **kwargs)

    def to_output(self, value):
        return ForecastOutput(predictions=value)


@register_head("point")
class PointForecastHead(ForecastHead):
    def __init__(
        self, prediction_length: int, target_dim: int = 1, residual: Optional[str] = None, version=2, **kwargs
    ):
        super().__init__(**kwargs)
        self.prediction_length = int(prediction_length)
        self.target_dim = int(target_dim)
        self.residual = residual
        self.version = version
        self.flatten = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(
            self.target_dim if version == 1 else self.prediction_length * self.target_dim
        )

    @classmethod
    def from_task_config(cls, config, **kwargs):
        return cls(
            config.output_chunk_length,
            config.target_dim,
            residual=config.residual,
            version=config.head_version,
            **kwargs,
        )

    def call(self, sequence_output, past_values=None, **kwargs):
        if self.version == 1:
            predictions = self.projection(sequence_output[:, -self.prediction_length :])
        else:
            predictions = tf.reshape(
                self.projection(self.flatten(sequence_output)), [-1, self.prediction_length, self.target_dim]
            )
        if self.residual is None:
            return predictions
        if past_values is None:
            raise ValueError("past_values is required for residual forecasting")
        tf.debugging.assert_equal(
            tf.shape(past_values)[-1], self.target_dim, message="Residual forecasts require exact target histories"
        )
        target = past_values
        if self.residual == "last_value":
            baseline = tf.tile(target[:, -1:, :], [1, self.prediction_length, 1])
        elif self.residual == "mean":
            baseline = tf.tile(tf.reduce_mean(target, axis=1, keepdims=True), [1, self.prediction_length, 1])
        elif self.residual == "last_window":
            baseline = target[:, -self.prediction_length :, :]
        else:
            raise ValueError("Unknown residual %r" % self.residual)
        return predictions + tf.cast(baseline, predictions.dtype)


@register_head("quantile")
class QuantileForecastHead(ForecastHead):
    def __init__(self, prediction_length: int, quantiles: Sequence[float], target_dim: int = 1, version=2, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = int(prediction_length)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.version = version
        self.target_dim = int(target_dim)
        if (
            not self.quantiles
            or tuple(sorted(set(self.quantiles))) != self.quantiles
            or any(q <= 0 or q >= 1 for q in self.quantiles)
        ):
            raise ValueError("quantiles must be unique, sorted, and between 0 and 1")
        self.flatten = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(
            (1 if version == 1 else self.prediction_length) * self.target_dim * len(self.quantiles)
        )

    @classmethod
    def from_task_config(cls, config, **kwargs):
        return cls(
            config.output_chunk_length,
            kwargs.pop("quantiles", config.quantiles),
            config.target_dim,
            version=config.head_version,
            **kwargs,
        )

    @property
    def default_objective(self):
        return ComponentSpec("quantile", {"quantiles": self.quantiles})

    def to_output(self, value):
        index = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        return ForecastOutput(predictions=value[..., index], quantile_values=value, quantiles=self.quantiles)

    def call(self, sequence_output, **kwargs):
        if self.version == 1:
            raw = self.projection(sequence_output[:, -self.prediction_length :])
            return tf.reshape(raw, tf.concat([tf.shape(raw)[:-1], [self.target_dim, len(self.quantiles)]], 0))
        values = tf.reshape(
            self.projection(self.flatten(sequence_output)),
            [-1, self.prediction_length, self.target_dim, len(self.quantiles)],
        )
        increments = tf.cumsum(tf.nn.softplus(values[..., 1:]), axis=-1)
        return values[..., :1] + tf.concat([tf.zeros_like(values[..., :1]), increments], axis=-1)


@register_head("distribution")
class DistributionForecastHead(ForecastHead):
    def __init__(self, distribution: DistributionOutput, prediction_length: int, version=2, **kwargs):
        super().__init__(**kwargs)
        self.distribution = distribution
        self.version = version
        self.prediction_length = int(prediction_length)
        self.flatten = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(self.prediction_length * 32)

    @classmethod
    def from_task_config(cls, config, **kwargs):
        return cls(NormalOutput(config.target_dim), config.output_chunk_length, version=config.head_version, **kwargs)

    @property
    def default_objective(self):
        return ComponentSpec("nll", {"distribution": self.distribution})

    def to_output(self, value):
        return ForecastOutput(predictions=self.distribution.mean(value), distribution_params=value)

    def call(self, sequence_output, **kwargs):
        if self.version == 1:
            return self.distribution.parameters(sequence_output[:, -self.prediction_length :])
        hidden = tf.reshape(self.projection(self.flatten(sequence_output)), [-1, self.prediction_length, 32])
        return self.distribution.parameters(hidden)


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
