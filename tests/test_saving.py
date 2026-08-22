import json
import os
import tempfile
import unittest
import zipfile

import numpy as np
import tensorflow as tf

from tfts.models.tcn import Encoder
from tfts.saving import get_custom_objects, load_model


class TestKerasModelLoading(unittest.TestCase):
    def test_discovers_tfts_objects_from_keras_archive(self):
        config = {
            "class_name": "Functional",
            "config": {
                "layers": [
                    {
                        "module": "tfts.models.tcn",
                        "class_name": "Encoder",
                        "registered_name": "Encoder",
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("config.json", json.dumps(config))

            custom_objects = get_custom_objects(model_path)

        self.assertIs(custom_objects["Encoder"], Encoder)

    def test_non_keras_archive_has_no_discovered_objects(self):
        with tempfile.NamedTemporaryFile() as model_file:
            self.assertEqual(get_custom_objects(model_file.name), {})

    def test_tcn_model_round_trip_without_custom_objects(self):
        inputs = tf.keras.Input(shape=(10, 1))
        outputs = Encoder(
            kernel_sizes=[2, 2],
            dilation_rates=[1, 2],
            filters=4,
            dense_hidden_size=3,
        )(
            inputs
        )[1]
        model = tf.keras.Model(inputs, outputs)
        sample = np.random.default_rng(1).normal(size=(2, 10, 1)).astype(np.float32)
        expected = model.predict(sample, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "tcn.keras")
            model.save(model_path)
            restored = load_model(model_path, compile=False)
            actual = restored.predict(sample, verbose=0)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
