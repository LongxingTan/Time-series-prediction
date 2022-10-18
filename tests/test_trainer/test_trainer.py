import unittest

import numpy as np
import tensorflow as tf

from tfts.models.auto_model import AutoModel
from tfts.trainer import KerasTrainer, Trainer


class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.fit_params = {
            "n_epochs": 3,
            "batch_size": 2,
        }

    def tearDown(self):
        pass

    def test_trainer_basic(self):
        # 1gpu, no dist
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
        valid_loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(1)

        for x, y in train_loader:
            self.assertEqual(x.shape, (1, 10, 1))
            break

        model = AutoModel("rnn", predict_length=2)
        trainer = Trainer(model)
        trainer.train(train_loader=train_loader, valid_loader=valid_loader, **self.fit_params)

    # def test_trainer_no_dist_strategy(self):
    #     pass
    #
    # def test_trainer_static_batch(self):
    #     pass
    #
    # def test_trainer_1gpu_with_dist_strategy(self):
    #     pass
    #
    # def test_trainer_2gpu(self):
    #     pass
    #
    # def test_trainer_fp16(self):
    #     pass
    #
    # def test_trainer_2gpu_fp16(self):
    #     pass
    #
    # def test_predict(self):
    #     pass
    #
    # def test_predict_fp16(self):
    #     pass


class KerasTrainerTest(unittest.TestCase):
    def setUp(self):
        self.fit_params = {
            "n_epochs": 3,
            "batch_size": 1,
        }

    def tearDown(self):
        pass

    def test_trainer_basic_array(self):
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        model = AutoModel("rnn", predict_length=2)

        trainer = KerasTrainer(model)
        trainer.train(train_dataset=(x_train, y_train), valid_dataset=(x_valid, y_valid), **self.fit_params)

    def test_trainer_basic_tfdata(self):
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
        valid_loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(1)

        model = AutoModel("rnn", predict_length=2)
        trainer = KerasTrainer(model)
        trainer.train(train_loader, valid_loader, **self.fit_params)
