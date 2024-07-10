import tensorflow as tf


class SegmentationHead(tf.keras.layers.Layer):
    def __init__(self):
        super(SegmentationHead, self).__init__()
