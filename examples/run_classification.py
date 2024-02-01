import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer


class CustomModel(tf.keras.Model):
    def __init__(
        self,
    ):
        super().__init__()

    def call(self):
        return
