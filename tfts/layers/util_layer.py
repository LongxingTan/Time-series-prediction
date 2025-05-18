import tensorflow as tf


class ShapeLayer(tf.keras.layers.Layer):
    """Layer to handle shape operations in a Keras-compatible way."""

    def __init__(self):
        super().__init__()

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        num_feature = tf.shape(x)[2]
        return batch_size, seq_length, num_feature
