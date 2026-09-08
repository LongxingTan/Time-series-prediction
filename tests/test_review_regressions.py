"""Guardrails for the independently reviewable model changes."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tfts.layers.autoformer_layer import AutoCorrelation
from tfts.models.autoformer import AutoFormer, AutoFormerConfig, DecoderLayer, EncoderLayer
from tfts.models.base import CommonConfig
from tfts.models.dlinear import DLinear, DLinearConfig
from tfts.models.itransformer import ITransformer, ITransformerConfig
from tfts.models.rwkv import RWKV, RWKVConfig
from tfts.models.tide import Tide, TideConfig
from tfts.training.saving import load_model


class ReviewRegressionTest(unittest.TestCase):
    def test_autocorrelation_dynamic_lengths(self):
        layer = AutoCorrelation(4, 2, factor=2)

        @tf.function(input_signature=[tf.TensorSpec([None, None, 4], tf.float32)])
        def forward(x):
            return layer(x, x, x)

        for length in (1, 7, 12):
            self.assertEqual(forward(tf.ones([2, length, 4])).shape, (2, length, 4))

    def test_target_validation_and_config_aliases(self):
        for cls in (AutoFormerConfig, TideConfig, DLinearConfig, ITransformerConfig, RWKVConfig):
            config = cls(target_dim=2, layer_norm_eps=0.002, d_model=32)
            self.assertEqual(config.target_dim, 2)
            self.assertEqual(config.layer_norm_eps, 0.002)
            self.assertEqual(config.hidden_size, 32)
        for model_cls, config_cls in ((AutoFormer, AutoFormerConfig), (Tide, TideConfig), (DLinear, DLinearConfig)):
            with self.subTest(model=model_cls.__name__):
                model = model_cls(3, config_cls(target_dim=2))
                with self.assertRaisesRegex(ValueError, "fewer channels"):
                    model.build((None, 12, 1))
                self.assertEqual(model(tf.ones([2, 12, 3])).shape, (2, 3, 2))
        self.assertEqual(CommonConfig().target_dim, 1)

    def test_autoformer_layer_config_and_activation_validation(self):
        settings = dict(d_model=8, d_ff=16, num_heads=2, moving_avg=3, act="gelu")
        for cls, kwargs in ((EncoderLayer, settings), (DecoderLayer, dict(settings, c_out=2))):
            layer = cls(**kwargs)
            rebuilt = cls.from_config(layer.get_config())
            self.assertEqual(rebuilt.get_config(), layer.get_config())
        with self.assertRaises(ValueError):
            AutoFormer(config=AutoFormerConfig(hidden_act="gellu"))

    def test_changed_models_save_and_fit(self):
        for cls, config in (
            (AutoFormer, AutoFormerConfig(hidden_size=8, num_attention_heads=2, num_layers=1)),
            (Tide, TideConfig(hidden_size=8, num_layers=1)),
            (ITransformer, ITransformerConfig(hidden_size=8, num_attention_heads=2, num_layers=1)),
        ):
            with self.subTest(model=cls.__name__):
                x = np.random.default_rng(7).normal(size=(4, 12, 1)).astype("float32")
                model = cls(3, config)
                model.compile(optimizer="adam", loss="mse", jit_compile=False)
                model.fit(x, x[:, :3], epochs=1, batch_size=4, verbose=0)
                expected = model.predict(x, verbose=0)
                self.assertEqual(expected.shape, (4, 3, 1))
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / "model.keras")
                    model.save(path)
                    restored = load_model(path, compile=False)
                    np.testing.assert_allclose(restored(x, training=False), model(x, training=False), atol=2e-5)
