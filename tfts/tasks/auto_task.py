"""Time series task head"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D

from .base import BaseTask


class PredictionHead(tf.keras.layers.Layer, BaseTask):
    """Prediction task head layer"""

    def __init__(self):
        super(PredictionHead, self).__init__()


class SegmentationHead(tf.keras.layers.Layer, BaseTask):
    """Segmentation task head layer"""

    def __init__(self):
        super(SegmentationHead, self).__init__()


class ClassificationHead(tf.keras.layers.Layer):
    """Classification task head layer"""

    def __init__(self, num_labels: int = 1, dense_units: Tuple[int] = (128,)):
        super(ClassificationHead, self).__init__()
        self.pooling = GlobalAveragePooling1D(data_format="channels_last")
        self.dense_units = dense_units
        self.dense = Dense(num_labels, activation="softmax")

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

        for unit in self.dense_units:
            pooled_output = Dense(unit, activation="relu")(pooled_output)

        # => (batch_size, num_labels)
        logits = self.dense(pooled_output)
        return logits


class AnomalyHead:
    """Anomaly task head layer: Reconstruct style"""

    def __init__(self, train_sequence_length: int) -> None:
        super().__init__()
        self.train_sequence_length = train_sequence_length

    def __call__(self, y_pred, y_test):
        if isinstance(y_pred, tf.Tensor):
            y_pred = y_pred.numpy()
        if y_pred.shape[1] == 1:
            y_pred = np.squeeze(y_pred, 1)
        errors = y_pred - y_test

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
