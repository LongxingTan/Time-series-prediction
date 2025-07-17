import tensorflow as tf


class ShapeLayer(tf.keras.layers.Layer):
    """Layer to handle shape operations in a Keras-compatible way."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.shape(x)


class CreateDecoderFeature(tf.keras.layers.Layer):
    def __init__(self, predict_sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.predict_sequence_length = predict_sequence_length

    def call(self, encoder_feature):
        batch_size = tf.shape(encoder_feature)[0]
        time_range = tf.range(self.predict_sequence_length)
        tiled = tf.tile(tf.reshape(time_range, (1, self.predict_sequence_length, 1)), (batch_size, 1, 1))
        return tf.cast(tiled, tf.float32)
