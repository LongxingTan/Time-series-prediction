"""Time series task head"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D


class PredictionHead(tf.keras.layers.Layer):
    """Prediction task head layer"""

    def __init__(self):
        super(PredictionHead, self).__init__()


class SegmentationHead(tf.keras.layers.Layer):
    """Segmentation task head layer"""

    def __init__(self):
        super(SegmentationHead, self).__init__()


class ClassificationHead(tf.keras.layers.Layer):
    """Classification task head layer"""

    def __init__(self, num_labels: int = 1):
        super(ClassificationHead, self).__init__()
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(num_labels)

    def call(self, inputs):
        """classification task head

        Parameters
        ----------
        inputs : tf.Tensor
            model backbone output as task input

        Returns
        -------
        tf.Tensor
            _description_
        """
        pooled_output = self.pooling(inputs)
        logits = self.dense(pooled_output)
        return logits


class AnomalyHead(tf.keras.layers.Layer):
    """Anomaly task head layer: Reconstruct style"""

    def __init__(self, train_sequence_length) -> None:
        super(AnomalyHead, self).__init__()
        self.train_sequence_length = train_sequence_length

    def call(self, y_pred, y_test):
        """anomaly task head

        Parameters
        ----------
        y_pred : tf.Tensor
            model predict
        y_test: tf.Tensor
            model truth

        Returns
        -------
        tf.Tensor
            distance
        """
        y_pred = y_pred.numpy()
        errors = y_pred - y_test

        mean = sum(errors) / len(errors)
        cov = 0
        for e in errors:
            cov += np.dot((e - mean).reshape(len(e), 1), (e - mean).reshape(1, len(e)))
        cov /= len(errors)

        m_dist = [0] * self.train_sequence_length
        for e in errors:
            m_dist.append(self._mahala_distance(e, mean, cov))

        return m_dist

    def _mahala_distance(self, x, mean, cov):
        """calculate Mahalanobis distance"""
        d = np.dot(x - mean, np.linalg.inv(cov))
        d = np.dot(d, (x - mean).T)
        return d


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

    def call(self, x):
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
