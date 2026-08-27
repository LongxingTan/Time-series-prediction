"""Anomaly scoring and calibration, deliberately separate from neural heads."""

from abc import ABC, abstractmethod

import tensorflow as tf

from .base import BaseTask


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
