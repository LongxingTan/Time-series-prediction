import tensorflow as tf


class ShapeLayer(tf.keras.layers.Layer):
    """Layer to handle shape operations in a Keras-compatible way."""

    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.shape(x)
