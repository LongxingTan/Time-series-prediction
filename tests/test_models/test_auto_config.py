import unittest

import tensorflow as tf

import tfts
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.models.registry import get_model_info, list_models


class TestAutoModel(unittest.TestCase):
    def test_auto_config(self):
        config = AutoConfig.for_model("bert")
        print(config.hidden_size)

    def test_top_level_import_exports_public_api(self):
        self.assertTrue(hasattr(tfts, "AutoConfig"))
        self.assertTrue(hasattr(tfts, "AutoModel"))
        self.assertIn("bert", tfts.list_models())

    def test_listed_models_have_auto_configs(self):
        for model_name in list_models():
            with self.subTest(model_name=model_name):
                config = AutoConfig.for_model(model_name)
                self.assertEqual(config.model_type, model_name)

    def test_listed_models_can_be_instantiated(self):
        for model_name in list_models():
            with self.subTest(model_name=model_name):
                config = AutoConfig.for_model(model_name)
                model = AutoModel.from_config(config, predict_sequence_length=2)
                self.assertEqual(model.config.model_type, model_name)

    def test_registry_metadata_matches_auto_dispatch(self):
        for model_name in list_models():
            with self.subTest(model_name=model_name):
                info = get_model_info(model_name)
                config = AutoConfig.for_model(model_name)
                model = AutoModel.from_config(config, predict_sequence_length=2)

                self.assertEqual(type(config).__name__, info["config_class"])
                self.assertEqual(type(model.model).__name__, info["class_name"])

    def test_listed_models_satisfy_forward_contract(self):
        # AutoFormer currently operates in its configured hidden dimension;
        # N-BEATS is intentionally univariate. Other models accept a small
        # multivariate input.
        feature_counts = {"autoformer": 64, "nbeats": 1}
        multivariate_outputs = {"diffusion", "itransformer", "timesnet", "timexer", "timemixer"}

        predict_sequence_length = 8
        for model_name in list_models():
            with self.subTest(model_name=model_name):
                feature_count = feature_counts.get(model_name, 3)
                config = AutoConfig.for_model(model_name)
                model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
                if model_name == "deep_ar":
                    inputs = {
                        "x": tf.random.normal([1, 16, 1]),
                        "decoder_feature": tf.random.normal([1, predict_sequence_length, 1]),
                        "static": tf.zeros([1, 1], dtype=tf.int32),
                    }
                else:
                    inputs = tf.random.normal([1, 16, feature_count])
                output = model(inputs)

                if model_name == "deep_ar":
                    self.assertEqual(len(output), 2)
                    self.assertEqual(output["loc"].shape, (1, predict_sequence_length, 1))
                    self.assertEqual(output["scale"].shape, (1, predict_sequence_length, 1))
                elif model_name in multivariate_outputs:
                    self.assertEqual(output.shape, (1, predict_sequence_length, feature_count))
                else:
                    self.assertEqual(output.shape, (1, predict_sequence_length, 1))
