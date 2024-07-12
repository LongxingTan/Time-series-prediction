import numpy as np
import tensorflow as tf


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

    def __init__(self):
        super(ClassificationHead, self).__init__()


class AnomalyHead(tf.keras.layers.Layer):
    """Anomaly task head layer: Reconstruct style"""

    def __init__(self, model, train_sequence_length) -> None:
        self.model = model
        self.train_sequence_length = train_sequence_length

    def detect(self, x_test, y_test):
        y_pred = self.model(x_test)
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
        # calculate Mahalanobis distance
        d = np.dot(x - mean, np.linalg.inv(cov))
        d = np.dot(d, (x - mean).T)
        return d
