import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts import AutoModel, KerasTrainer


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.test_models = ["seq2seq", "wavenet", "transformer", "rnn", "tcn", "bert", "informer"]

    def test_encoder_array(self):
        train_length = 49
        predict_length = 10
        n_feature = 2
        x_train = np.random.rand(1, train_length, n_feature)
        y_train = np.random.rand(1, predict_length, 1)
        x_valid = np.random.rand(1, train_length, n_feature)
        y_valid = np.random.rand(1, predict_length, 1)

        for m in self.test_models:
            model = AutoModel(m, predict_length=predict_length)
            trainer = KerasTrainer(model)
            trainer.train(train_dataset=(x_train, y_train), valid_dataset=(x_valid, y_valid), n_epochs=1)

    def test_encoder_decoder_array(self):
        train_length = 49
        predict_length = 10
        n_encoder_feature = 2
        n_decoder_feature = 3
        x_train = {
            "x": np.random.rand(1, train_length, 1),
            "encoder_feature": np.random.rand(1, train_length, n_encoder_feature),
            "decoder_feature": np.random.rand(1, predict_length, n_decoder_feature),
        }
        y_train = np.random.rand(1, predict_length, 1)
        x_valid = {
            "x": np.random.rand(1, train_length, 1),
            "encoder_feature": np.random.rand(1, train_length, n_encoder_feature),
            "decoder_feature": np.random.rand(1, predict_length, n_decoder_feature),
        }
        y_valid = np.random.rand(1, predict_length, 1)

        for m in self.test_models:
            model = AutoModel(m, predict_length=predict_length)
            trainer = KerasTrainer(model)
            trainer.train((x_train, y_train), (x_valid, y_valid), n_epochs=1)

    def test_encoder_decoder_array2(self):
        train_length = 49
        predict_length = 10
        n_encoder_feature = 2
        n_decoder_feature = 3

        x_train = (
            np.random.rand(1, train_length, 1),
            np.random.rand(1, train_length, n_encoder_feature),
            np.random.rand(1, predict_length, n_decoder_feature),
        )
        y_train = np.random.rand(1, predict_length, 1)
        x_valid = (
            np.random.rand(1, train_length, 1),
            np.random.rand(1, train_length, n_encoder_feature),
            np.random.rand(1, predict_length, n_decoder_feature),
        )
        y_valid = np.random.rand(1, predict_length, 1)

        for m in self.test_models:
            model = AutoModel(m, predict_length=predict_length)
            trainer = KerasTrainer(model)
            trainer.train((x_train, y_train), (x_valid, y_valid), n_epochs=1)

    # def test_encoder_tfdata(self):
    #     train_length = 20
    #     predict_length = 10
    #     n_feature = 2
    #
    #     x_train = np.random.rand(1, train_length, n_feature)
    #     y_train = np.random.rand(1, predict_length, 1)
    #     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=1)
    #     x_valid = np.random.rand(1, train_length, n_feature)
    #     y_valid = np.random.rand(1, predict_length, 1)
    #     valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=1)
    #
    #     for m in self.test_models:
    #         model = AutoModel(m, predict_length=predict_length)
    #         trainer = KerasTrainer(model)
    #         trainer.train(train_dataset=train_dataset, valid_dataset=valid_dataset, n_epochs=1)

    def test_encoder_decoder_tfdata(self):
        predict_length = 10
        train_reader = FakeReader(predict_length=predict_length)
        train_loader = tf.data.Dataset.from_generator(
            train_reader.iter,
            ({"x": tf.float32, "encoder_feature": tf.float32, "decoder_feature": tf.float32}, tf.float32),
        )
        train_loader = train_loader.batch(batch_size=1)
        valid_reader = FakeReader(predict_length=predict_length)
        valid_loader = tf.data.Dataset.from_generator(
            valid_reader.iter,
            ({"x": tf.float32, "encoder_feature": tf.float32, "decoder_feature": tf.float32}, tf.float32),
        )
        valid_loader = valid_loader.batch(batch_size=1)

        for m in self.test_models:
            model = AutoModel(m, predict_length=predict_length)
            trainer = KerasTrainer(model)
            trainer.train(train_dataset=train_loader, valid_dataset=valid_loader, n_epochs=1)


class FakeReader(object):
    def __init__(self, predict_length=10):
        train_length = 20
        n_encoder_feature = 2
        n_decoder_feature = 3
        self.x = np.random.rand(5, train_length, 1)
        self.encoder_feature = np.random.rand(5, train_length, n_encoder_feature)
        self.decoder_feature = np.random.rand(5, predict_length, n_decoder_feature)
        self.target = np.random.rand(5, predict_length, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "encoder_feature": self.encoder_feature[idx],
            "decoder_feature": self.decoder_feature[idx],
        }, self.target[idx]

    def iter(self):
        for i in range(len(self.x)):
            yield self[i]
