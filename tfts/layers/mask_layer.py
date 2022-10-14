# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.transformer`"""

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers


class MaskLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
