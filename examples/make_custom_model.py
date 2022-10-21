"""How to define custom model from tfts"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

from tfts import AutoConfig, AutoModel


def build_model():
    train_length = 24
    train_features = 15
    predict_length = 16

    inputs = Input([train_length, train_features])
    backbone = AutoModel("seq2seq", predict_length=predict_length)
    outputs = backbone(inputs)
    outputs = Dense(1, activation="sigmoid")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer="rmsprop")
    return model
