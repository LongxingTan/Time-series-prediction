import os
import shutil
import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModel
from tfts.trainer import KerasTrainer, Trainer


class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.fit_config = {
            "epochs": 2,
            "stop_no_improve_epochs": 1,
            "eval_metric": lambda x, y: np.mean(np.abs(x.numpy() - y.numpy())),
            "model_dir": "./weights",
        }

        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        self.train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
        self.valid_loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(1)

        for x, y in self.train_loader:
            self.assertEqual(x.shape, (1, 10, 1))
            break

    def tearDown(self):
        if os.path.exists("./weights"):
            shutil.rmtree("./weights", ignore_errors=True)

    def test_trainer_basic(self):
        # 1gpu, no dist
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model, optimizer=tf.keras.optimizers.legacy.Adam(0.003))
        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, **self.fit_config)
        trainer.predict(self.valid_loader)
        trainer.save_model(model_dir="./weights", only_pb=False)

    # def test_trainer_no_dist_strategy(self):
    #     pass
    #
    # def test_trainer_static_batch(self):
    #     pass
    #
    # def test_trainer_1gpu_with_dist_strategy(self):
    #     pass

    def test_trainer_2gpu(self):
        strategy = tf.distribute.MirroredStrategy()
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model, strategy=strategy)
        trainer.train(self.train_loader, self.valid_loader, **self.fit_config)

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
        self.fit_config = {
            "epochs": 1,
            "batch_size": 1,
        }

    def tearDown(self):
        if os.path.exists("./weights"):
            shutil.rmtree("./weights", ignore_errors=True)

    def test_trainer_basic_array(self):
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        trainer = KerasTrainer(model, optimizer=tf.keras.optimizers.legacy.Adam(0.003))
        trainer.train(train_dataset=(x_train, y_train), valid_dataset=(x_valid, y_valid), **self.fit_config)
        y_valid_pred = trainer.predict(x_valid)
        self.assertEqual(y_valid_pred.shape, (1, 2, 1))

    def test_trainer_basic_tfdata(self):
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
        valid_loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(1)

        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model, optimizer=tf.keras.optimizers.legacy.Adam(0.003))
        trainer.train(train_loader, valid_loader, **self.fit_config)
        trainer.save_model("./weights")
