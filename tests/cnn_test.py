import sys
import os
filePath = os.path.abspath(os.path.dirname(''))
sys.path.append(os.path.split(filePath)[0])

import pytest
from deepts.models.cnn import CNN
from examples.data.load_data import DataLoader
import tensorflow as tf


def test_cnn_shape():
    import numpy as np
    fake_data = np.random.rand(16, 160, 35)
    cnn = CNN(custom_model_params={})
    y = cnn(tf.convert_to_tensor(fake_data, tf.float32))
    print(y.shape)


def test_cnn_model():
    data_loader = DataLoader('sine')
    dataset = data_loader(params={}, data_dir=None, batch_size=8, training=True)
    print(dataset.take(1))

    inputs = tf.keras.layers.Input([30, 2])
    cnn_model = CNN()
    model = tf.keras.Model(inputs, cnn_model(inputs))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=3e-4))
    model.fit(dataset, epochs=5)


if __name__ == '__main__':
    test_cnn_shape()
