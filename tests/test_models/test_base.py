import os
import tempfile
import unittest

import tensorflow as tf

from tfts.models.base import BaseConfig, BaseModel


class TestBaseConfig(unittest.TestCase):
    def setUp(self):
        self.config_data = {"input_shape": [100], "hidden_size": 128, "num_layers": 2}
        self.config = BaseConfig(**self.config_data)

    def test_config_initialization(self):
        self.assertEqual(self.config.input_shape, [100])
        self.assertEqual(self.config.hidden_size, 128)
        self.assertEqual(self.config.num_layers, 2)

    def test_config_to_dict(self):
        config_dict = self.config.to_dict()
        self.assertEqual(config_dict, self.config_data)

    def test_config_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_path = os.path.join(tmpdirname, "config.json")
            self.config.to_json(config_path)
            loaded_config = BaseConfig.from_json(config_path)
            self.assertEqual(loaded_config.to_dict(), self.config.to_dict())


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.config_data = {"input_shape": [100], "hidden_size": 128, "num_layers": 2}
        self.config = BaseConfig(**self.config_data)
        self.model = BaseModel(config=self.config)

    def test_model_initialization(self):
        self.assertEqual(self.model.config.input_shape, [100])
        self.assertEqual(self.model.predict_sequence_length, 1)

    # def test_model_save_and_load(self):
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         weights_dir = os.path.join(tmpdirname, "weights.h5")
    #         config_dir = os.path.join(tmpdirname, "config.json")
    #         self.model.save_weights(tmpdirname)
    #         self.assertTrue(os.path.exists(weights_dir))
    #         self.assertTrue(os.path.exists(config_dir))
    #
    #         new_model = BaseModel.from_pretrained(tmpdirname)
    #         self.assertEqual(new_model.config.to_dict(), self.config.to_dict())
