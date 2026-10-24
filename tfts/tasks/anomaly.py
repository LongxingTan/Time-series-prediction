from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from tfts.forecast import ForecastOutput
from tfts.models.base import BackboneOutput, ModelOutput, OutputPort
from tfts.registry import ComponentSpec, register_task, resolve

from .auto_task import ClassificationHead, ReconstructionHead
from .base import BaseTask, TaskConfig, TaskType, TimeSeriesTaskModel


class AnomalyScorer(BaseTask, ABC):
    @abstractmethod
    def __call__(self, observed, reconstructed, observed_mask=None):
        raise NotImplementedError


class SquaredErrorScorer(AnomalyScorer):
    def __call__(self, observed, reconstructed, observed_mask=None):
        error = tf.square(tf.cast(observed, reconstructed.dtype) - reconstructed)
        if observed_mask is not None:
            error = error * tf.cast(observed_mask, error.dtype)
        return tf.reduce_mean(error, axis=-1)


class AbsoluteErrorScorer(AnomalyScorer):
    def __call__(self, observed, reconstructed, observed_mask=None):
        error = tf.abs(tf.cast(observed, reconstructed.dtype) - reconstructed)
        if observed_mask is not None:
            error = error * tf.cast(observed_mask, error.dtype)
        return tf.reduce_mean(error, axis=-1)


class QuantileCalibrator(tf.Module):
    """Fit a reproducible threshold from normal/calibration scores only."""

    def __init__(self, quantile=0.99, name=None):
        super().__init__(name=name)
        if not 0 < quantile < 1:
            raise ValueError("quantile must lie strictly between 0 and 1")
        self.quantile = float(quantile)
        self.threshold = tf.Variable(float("nan"), trainable=False, dtype=tf.float32)
        self.fitted = tf.Variable(False, trainable=False, dtype=tf.bool)

    def fit(self, scores):
        values = tf.sort(tf.reshape(tf.cast(scores, tf.float32), [-1]))
        count = tf.size(values)
        tf.debugging.assert_positive(count, message="Cannot calibrate from empty scores")
        index = tf.cast(tf.math.ceil(self.quantile * tf.cast(count, tf.float32)) - 1, tf.int32)
        index = tf.clip_by_value(index, 0, count - 1)
        self.threshold.assign(values[index])
        self.fitted.assign(True)
        return self.threshold

    def predict(self, scores):
        tf.debugging.assert_equal(self.fitted, True, message="Calibrator must be fitted before prediction")
        return tf.cast(scores > tf.cast(self.threshold, scores.dtype), tf.int32)


def make_anomaly_scorer(name):
    scorers = {"squared_error": SquaredErrorScorer, "absolute_error": AbsoluteErrorScorer}
    try:
        return scorers[name]()
    except KeyError as error:
        raise ValueError("Unknown anomaly scorer %r. Available: %s" % (name, sorted(scorers))) from error


@dataclass(frozen=True)
class AnomalyDetectionTaskConfig(TaskConfig):
    task: TaskType = TaskType.ANOMALY_DETECTION
    target_dim: int = 1
    scorer: str = "squared_error"
    threshold_quantile: float = 0.99

    def __post_init__(self):
        super().__post_init__()
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive")
        if self.scorer not in {"squared_error", "absolute_error"}:
            raise ValueError("Unknown anomaly scorer %r" % self.scorer)
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must lie strictly between 0 and 1")


@dataclass(frozen=True)
class AnomalyDetectionOutput(ModelOutput):
    reconstruction: Optional[tf.Tensor] = None
    scores: Optional[tf.Tensor] = None
    labels: Optional[tf.Tensor] = None
    threshold: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None

    def prediction(self):
        """Return the task's primary prediction tensor."""
        return self.scores


@register_task("anomaly_detection", config_class=AnomalyDetectionTaskConfig)
@tf.keras.utils.register_keras_serializable(package="tfts")
class AnomalyDetectionModel(TimeSeriesTaskModel):
    task_name = "anomaly_detection"
    default_target_field = "past_values"
    required_output_port = OutputPort.TEMPORAL_SEQUENCE
    default_objective = "masked_mse"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        if not capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
            raise ValueError(f"{backbone.config.model_type} does not support anomaly detection")
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ReconstructionHead(task_config.target_dim)
        self.scorer = make_anomaly_scorer(task_config.scorer)
        self.calibrator = QuantileCalibrator(task_config.threshold_quantile)
        objective = objective or ComponentSpec(
            "masked_mse",
            {"selection": "observed", "target_field": "labels", "prediction_field": "reconstruction"},
        )
        self.configure_objective(objective)

    def forward(self, inputs, training=None):
        batch = self.normalize_batch(inputs)
        backbone_output = self.adapter.forward(batch, training=training, require=OutputPort.TEMPORAL_SEQUENCE)
        temporal = backbone_output.sequence_output[:, : tf.shape(batch.past_values)[1], :]
        reconstruction = self.head(temporal)
        scores = self.scorer(batch.past_values, reconstruction, batch.past_observed_mask)
        return AnomalyDetectionOutput(
            reconstruction=reconstruction,
            scores=scores,
            backbone_output=backbone_output,
        )

    def calibrate(self, inputs):
        return self.calibrator.fit(self.forward(inputs, training=False).scores)

    def detect(self, inputs):
        output = self.forward(inputs, training=False)
        return output.replace(
            labels=self.calibrator.predict(output.scores),
            threshold=tf.identity(self.calibrator.threshold),
        )
