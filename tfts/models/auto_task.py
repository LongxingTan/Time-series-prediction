"""Time series task head"""

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
