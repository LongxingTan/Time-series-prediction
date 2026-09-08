"""Registry-wide forecast shape and Keras lifecycle contracts."""

import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.models.registry import get_model_capabilities, list_models


class RegistryShapeTest(unittest.TestCase):
    def test_registered_sequence_models_fit_predict(self):
        x = np.random.default_rng(5).normal(size=(2, 24, 1)).astype("float32")
        y = x[:, :3]
        for name in list_models():
            with self.subTest(model=name):
                tf.keras.backend.clear_session()
                capabilities = get_model_capabilities(name)
                config = AutoConfig.for_model(name)
                config.update(dict(hidden_size=16, num_layers=1, num_attention_heads=2))
                model = AutoModelForForecasting.from_config(config, output_chunk_length=3, target_dim=1)
                if capabilities.input_spec.arrangement.value != "none":
                    # Spatial models explicitly reject the (B,T,C) contract;
                    # their node-aware fit/shape tests live in test_spatial_models.
                    with self.assertRaises(ValueError):
                        model(x)
                    continue
                self.assertEqual(model(x, training=False).predictions.shape, (2, 3, 1))
                model.compile(optimizer="adam", jit_compile=False)
                model.fit(x, y, batch_size=2, epochs=1, verbose=0)
                prediction = model.predict(x, batch_size=2, verbose=0)
                self.assertEqual(prediction.shape, (2, 3, 1))
