import unittest

import tfts
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.models.registry import list_models


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
