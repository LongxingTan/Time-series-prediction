import logging
import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModel, KerasTrainer
from tfts.training import TrainingArguments

logger = logging.getLogger(__name__)

# Smoke test pinning a single-device strategy so it runs identically on CI and
# any multi-GPU host.
_SINGLE_DEVICE_ARGS = TrainingArguments(output_dir="./weights", strategy="default")


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.test_models = ["seq2seq", "wavenet", "transformer", "rnn", "tcn", "bert", "informer"]

    def test_encoder_array(self):
        train_length = 32
        predict_sequence_length = 9
        n_feature = 2
        x_train = np.random.rand(1, train_length, n_feature)
        y_train = np.random.rand(1, predict_sequence_length, 1)
        x_valid = np.random.rand(1, train_length, n_feature)
        y_valid = np.random.rand(1, predict_sequence_length, 1)

        for m in self.test_models:
            print(f"==== Test model {m} ====")
            config = AutoConfig.for_model(m)
            model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
            trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
            trainer.train(
                train_dataset=(x_train, y_train),
                valid_dataset=(x_valid, y_valid),
                optimizer=tf.keras.optimizers.Adam(0.003),
                epochs=1,
            )

    def test_encoder_decoder_array(self):
        train_length = 32
        predict_sequence_length = 9
        n_encoder_feature = 2
        n_decoder_feature = 3
        x_train = {
            "past_values": np.random.rand(1, train_length, 1),
            "past_time_features": np.random.rand(1, train_length, n_encoder_feature),
            "future_time_features": np.random.rand(1, predict_sequence_length, n_decoder_feature),
        }
        y_train = np.random.rand(1, predict_sequence_length, 1)
        x_valid = {
            "past_values": np.random.rand(1, train_length, 1),
            "past_time_features": np.random.rand(1, train_length, n_encoder_feature),
            "future_time_features": np.random.rand(1, predict_sequence_length, n_decoder_feature),
        }
        y_valid = np.random.rand(1, predict_sequence_length, 1)

        for m in self.test_models:
            config = AutoConfig.for_model(m)
            model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
            trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
            trainer.train((x_train, y_train), (x_valid, y_valid), optimizer=tf.keras.optimizers.Adam(0.003), epochs=1)

    def test_encoder_decoder_array2(self):
        train_length = 32
        predict_sequence_length = 9
        n_encoder_feature = 2
        n_decoder_feature = 3

        x_train = {
            "past_values": np.random.rand(1, train_length, 1),
            "past_time_features": np.random.rand(1, train_length, n_encoder_feature),
            "future_time_features": np.random.rand(1, predict_sequence_length, n_decoder_feature),
        }
        y_train = np.random.rand(1, predict_sequence_length, 1)
        x_valid = {
            "past_values": np.random.rand(1, train_length, 1),
            "past_time_features": np.random.rand(1, train_length, n_encoder_feature),
            "future_time_features": np.random.rand(1, predict_sequence_length, n_decoder_feature),
        }
        y_valid = np.random.rand(1, predict_sequence_length, 1)

        for m in self.test_models:
            print(f"==== Test model {m} ====")
            config = AutoConfig.for_model(m)
            model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
            trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
            trainer.train((x_train, y_train), (x_valid, y_valid), optimizer=tf.keras.optimizers.Adam(0.003), epochs=1)

    # def test_encoder_tfdata(self):
    #     train_length = 20
    #     predict_sequence_length = 10
    #     n_feature = 2
    #
    #     x_train = np.random.rand(1, train_length, n_feature)
    #     y_train = np.random.rand(1, predict_sequence_length, 1)
    #     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=1)
    #     x_valid = np.random.rand(1, train_length, n_feature)
    #     y_valid = np.random.rand(1, predict_sequence_length, 1)
    #     valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=1)
    #
    #     for m in self.test_models:
    #         model = AutoModel(m, predict_sequence_length=predict_sequence_length)
    #         trainer = KerasTrainer(model, optimizer=tf.keras.optimizers.legacy.Adam(0.003))
    #         trainer.train(train_dataset=train_dataset, valid_dataset=valid_dataset, epochs=1)

    def test_encoder_decoder_tfdata(self):
        predict_sequence_length = 10
        train_reader = FakeReader(predict_sequence_length=predict_sequence_length)
        train_loader = tf.data.Dataset.from_generator(
            train_reader.iter,
            output_signature=(
                {
                    "past_values": tf.TensorSpec([20, 1], tf.float32),
                    "past_time_features": tf.TensorSpec([20, 2], tf.float32),
                    "future_time_features": tf.TensorSpec([predict_sequence_length, 3], tf.float32),
                },
                tf.TensorSpec([predict_sequence_length, 1], tf.float32),
            ),
        )
        train_loader = train_loader.batch(batch_size=1)
        valid_reader = FakeReader(predict_sequence_length=predict_sequence_length)
        valid_loader = tf.data.Dataset.from_generator(
            valid_reader.iter,
            output_signature=(
                {
                    "past_values": tf.TensorSpec([20, 1], tf.float32),
                    "past_time_features": tf.TensorSpec([20, 2], tf.float32),
                    "future_time_features": tf.TensorSpec([predict_sequence_length, 3], tf.float32),
                },
                tf.TensorSpec([predict_sequence_length, 1], tf.float32),
            ),
        )
        valid_loader = valid_loader.batch(batch_size=1)

        for m in self.test_models:
            print(f"==== Test model {m} ====")
            config = AutoConfig.for_model(m)
            model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
            trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
            trainer.train(
                train_dataset=train_loader,
                valid_dataset=valid_loader,
                optimizer=tf.keras.optimizers.Adam(0.003),
                epochs=1,
            )


class FakeReader(object):
    def __init__(self, predict_sequence_length=10):
        train_length = 20
        n_encoder_feature = 2
        n_decoder_feature = 3
        self.x = np.random.rand(5, train_length, 1)
        self.encoder_feature = np.random.rand(5, train_length, n_encoder_feature)
        self.decoder_feature = np.random.rand(5, predict_sequence_length, n_decoder_feature)
        self.target = np.random.rand(5, predict_sequence_length, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "past_values": self.x[idx].astype(np.float32),
            "past_time_features": self.encoder_feature[idx].astype(np.float32),
            "future_time_features": self.decoder_feature[idx].astype(np.float32),
        }, self.target[idx].astype(np.float32)

    def iter(self):
        for i in range(len(self.x)):
            yield self[i]
