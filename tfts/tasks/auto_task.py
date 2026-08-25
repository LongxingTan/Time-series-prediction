"""Time series task head"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D

from tfts.distributions import DistributionOutput

from .base import BaseHead, BaseTask, ModelOutput


def apply_prediction_residual(
    outputs: tf.Tensor, inputs: tf.Tensor, residual: Optional[str], num_labels: int = 1
) -> tf.Tensor:
    """Apply the canonical prediction residual to already-projected outputs."""
    if residual is None:
        return outputs
    if residual not in PredictionHead._RESIDUALS:
        raise ValueError(f"Unknown prediction residual {residual!r}")
    if isinstance(inputs, dict):
        inputs = inputs["x"]
    elif isinstance(inputs, (list, tuple)):
        inputs = inputs[0]
    tf.debugging.assert_greater_equal(
        tf.shape(inputs)[-1], num_labels, message="Input has fewer channels than num_labels"
    )
    target = inputs[..., :num_labels]
    horizon = tf.shape(outputs)[1]
    if residual == "mean":
        value = tf.reduce_mean(target, axis=1, keepdims=True)
        residual_values = tf.tile(value, [1, horizon, 1])
    elif residual == "last_value":
        residual_values = tf.tile(target[:, -1:, :], [1, horizon, 1])
    else:
        tf.debugging.assert_greater_equal(
            tf.shape(target)[1], horizon, message="Input sequence is shorter than the prediction horizon"
        )
        residual_values = target[:, -horizon:, :]
    return outputs + tf.cast(residual_values, outputs.dtype)


class PredictionHead(BaseHead):
    """Project the final horizon of backbone states to a point forecast."""

    _RESIDUALS = {None, "last_value", "last_window", "mean"}

    def __init__(
        self,
        predict_sequence_length: Optional[int] = None,
        num_labels: int = 1,
        residual: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if predict_sequence_length is not None and predict_sequence_length <= 0:
            raise ValueError("predict_sequence_length must be positive")
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        if residual not in self._RESIDUALS:
            raise ValueError(f"residual must be one of {sorted(str(x) for x in self._RESIDUALS)}")
        self.predict_sequence_length = predict_sequence_length
        self.num_labels = num_labels
        self.residual = residual
        self.projection = Dense(num_labels, name="output_projection")

    def call(self, hidden_states: tf.Tensor, inputs: Optional[tf.Tensor] = None, **kwargs) -> tf.Tensor:
        if hidden_states.shape.rank != 3:
            raise ValueError("PredictionHead expects hidden states shaped (batch, time, hidden_size)")
        if self.predict_sequence_length is not None:
            tf.debugging.assert_greater_equal(
                tf.shape(hidden_states)[1],
                self.predict_sequence_length,
                message="Backbone returned fewer hidden states than the prediction horizon",
            )
            hidden_states = hidden_states[:, -self.predict_sequence_length :, :]
        outputs = self.projection(hidden_states)
        if self.residual is not None:
            if inputs is None:
                raise ValueError("inputs are required when PredictionHead.residual is enabled")
            outputs = apply_prediction_residual(outputs, inputs, self.residual, self.num_labels)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "predict_sequence_length": self.predict_sequence_length,
                "num_labels": self.num_labels,
                "residual": self.residual,
            }
        )
        return config


@dataclass
class PredictionOutput(ModelOutput):
    prediction_logits: tf.Tensor = None
    last_hidden_state: Optional[tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
    attentions: Optional[Tuple[tf.Tensor, ...]] = None
    loss: Optional[tf.Tensor] = None


class ClassificationHead(BaseHead):
    """Classification task head layer"""

    def __init__(self, num_labels: int = 1, dense_units: Tuple[int] = (128,)):
        super(ClassificationHead, self).__init__()
        self.pooling = GlobalAveragePooling1D(data_format="channels_last")
        self.intermediate_dense_layers = []
        for unit in dense_units:
            self.intermediate_dense_layers.append(Dense(unit, activation="relu"))
        self.classifier = Dense(num_labels, activation="softmax")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """classification task head

        Parameters
        ----------
        inputs : tf.Tensor
            model backbone output as task input, (batch_size, train_sequence_length, hidden_size)

        Returns
        -------
        tf.Tensor
            logit of the classification
        """
        # => (batch_size, hidden_size)
        pooled_output = self.pooling(inputs)

        for layer in self.intermediate_dense_layers:
            pooled_output = layer(pooled_output)

        # => (batch_size, num_labels)
        logits = self.classifier(pooled_output)
        return logits


@dataclass
class ClassificationOutput(ModelOutput):
    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
    loss: Optional[tf.Tensor] = None


class QuantileHead(BaseHead):
    """Project hidden states to ``(batch, time, labels, quantiles)``."""

    def __init__(self, quantiles: Sequence[float], num_labels: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.num_labels = int(num_labels)
        if not self.quantiles or any(q <= 0 or q >= 1 for q in self.quantiles):
            raise ValueError("quantiles must contain values strictly between 0 and 1")
        if tuple(sorted(self.quantiles)) != self.quantiles or len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError("quantiles must be unique and sorted")
        if self.num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.projection = Dense(self.num_labels * len(self.quantiles))

    def call(self, hidden_states: tf.Tensor, **kwargs) -> tf.Tensor:
        projected = self.projection(hidden_states)
        shape = tf.concat([tf.shape(projected)[:-1], [self.num_labels, len(self.quantiles)]], axis=0)
        return tf.reshape(projected, shape)


class DistributionHead(BaseHead):
    """Adapt a :class:`DistributionOutput` to the common task-head contract."""

    def __init__(self, distribution: DistributionOutput, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(distribution, DistributionOutput):
            raise TypeError("distribution must be a DistributionOutput instance")
        self.distribution = distribution

    def build(self, input_shape):
        # Parameter layers owned by DistributionOutput build on their first
        # ``parameters`` call; this head itself has no additional state.
        super().build(input_shape)

    def call(self, hidden_states: tf.Tensor, **kwargs):
        return self.distribution.parameters(hidden_states)


class AnomalyHead:
    """Anomaly task head layer: Reconstruct style"""

    def __init__(self, train_sequence_length: int) -> None:
        super().__init__()
        self.train_sequence_length = train_sequence_length

    def __call__(self, y_pred, y_test):
        if isinstance(y_pred, tf.Tensor):
            y_pred = y_pred.numpy()
        if isinstance(y_test, tf.Tensor):
            y_test = y_test.numpy()
        if y_pred.shape[1] == 1:
            y_pred = np.squeeze(y_pred, 1)
        errors = y_pred - y_test

        if errors.ndim == 3:
            # Flatten batch and sequence dimensions while keeping features
            # (Batch, Time, Features) -> (Batch * Time, Features)
            errors = errors.reshape(-1, errors.shape[-1])

        # mean / cov
        mean = sum(errors) / len(errors)
        cov = 0
        for e in errors:
            cov += np.dot((e - mean).reshape(len(e), 1), (e - mean).reshape(1, len(e)))
        cov /= len(errors)

        m_dist = [0] * self.train_sequence_length
        for e in errors:
            m_dist.append(AnomalyHead.mahala_distantce(e, mean, cov))

        return m_dist

    @staticmethod
    def mahala_distantce(x, mean, cov, epsilon=1e-8):
        cov += epsilon * np.eye(cov.shape[0])  # Zero Covariance
        d = np.dot(x - mean, np.linalg.inv(cov))
        d = np.dot(d, (x - mean).T)
        return d


@dataclass
class AnomalyOutput(ModelOutput):
    anomaly_scores: tf.Tensor = None
    reconstruction_logits: Optional[tf.Tensor] = None
    loss: Optional[tf.Tensor] = None


class GaussianHead(tf.keras.layers.Layer):
    def __init__(self, units: int):
        self.units = units
        super().__init__()

    def build(self, input_shape: Tuple[Optional[int], ...]):
        in_channels = input_shape[2]
        self.weight1 = self.add_weight(
            name="gauss_w1", shape=(in_channels, self.units), initializer=tf.keras.initializers.GlorotNormal()
        )
        self.weight2 = self.add_weight(
            name="gauss_w2", shape=(in_channels, self.units), initializer=tf.keras.initializers.GlorotNormal()
        )
        self.bias1 = self.add_weight(name="gauss_b1", shape=(self.units,), initializer=tf.keras.initializers.Zeros())
        self.bias2 = self.add_weight(name="gauss_b2", shape=(self.units,), initializer=tf.keras.initializers.Zeros())
        super().build(input_shape)

    def call(self, x: tf.Tensor):
        """Returns mean and standard deviation tensors.

        Args:
          x (tf.Tensor): Input tensor.

        Returns:
          Tuple[tf.Tensor, tf.Tensor]: Mean and standard deviation tensors.
        """
        mu = tf.matmul(x, self.weight1) + self.bias1
        sig = tf.matmul(x, self.weight2) + self.bias2
        sig_pos = tf.math.log1p(tf.math.exp(sig)) + 1e-7
        return mu, sig_pos

    def get_config(self):
        """Returns the configuration of the layer."""
        config = {"units": self.units}
        base_config = super().get_config()
        return {**base_config, **config}


class SegmentationHead(tf.keras.layers.Layer, BaseTask):
    """Segmentation task head layer"""

    def __init__(self):
        super(SegmentationHead, self).__init__()
