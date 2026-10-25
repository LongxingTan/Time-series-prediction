"""Execute documentation snippets that cross the public model/output boundary."""

from pathlib import Path
import re
import tempfile
import textwrap
import unittest

import numpy as np
import tensorflow as tf

_ROOT = Path(__file__).resolve().parents[2]


def snippet(path, marker):
    source = (_ROOT / path).read_text(encoding="utf-8")
    blocks = re.findall(r"\.\. code-block:: python\n((?:[ \t].*\n|\n)+)", source)
    return next(textwrap.dedent(block).strip() for block in blocks if marker in block)


class DocumentationTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()

    def test_functional_forecast_wrappers(self):
        for path, marker, name, horizon in (
            ("docs/source/tutorials.rst", "class ForecastLayer", "keras_model", 8),
            ("docs/source/faq.rst", "class ForecastLayer", "model", 24),
        ):
            with self.subTest(path=path):
                namespace = {}
                exec(snippet(path, marker), namespace)
                model = namespace[name]
                self.assertEqual(model(tf.ones([2, 24, 1])).shape, (2, horizon, 1))
                self.assertTrue(np.isfinite(model.train_on_batch(tf.ones([2, 24, 1]), tf.zeros([2, horizon, 1]))).all())

    def test_custom_heads_consume_forecast_tensors(self):
        for path in ("docs/source/models.rst", "docs/source/faq.rst"):
            with self.subTest(path=path):
                namespace = {}
                exec(snippet(path, "class CustomModel"), namespace)
                cls = namespace["CustomModel"]
                model = cls(24) if path.endswith("models.rst") else cls()
                self.assertEqual(model(tf.ones([2, 24, 1])).shape, (2, 24, 1))

    def test_deployment_builds_before_saving(self):
        code = snippet("docs/source/tutorials.rst", "# Alternatively, export")
        with tempfile.TemporaryDirectory() as directory:
            namespace = {}
            exec(code.replace('"./my_model"', repr(directory)), namespace)
            from tfts import AutoModel

            restored = AutoModel.from_pretrained(directory)
            self.assertEqual(restored(tf.ones([2, 24, 1])).predictions.shape, (2, 8, 1))

    def test_serving_export(self):
        from tfts import AutoConfig, AutoModelForForecasting

        model = AutoModelForForecasting.from_config(AutoConfig.for_model("dlinear"), output_chunk_length=8)
        inputs = tf.ones([2, 24, 1])
        expected = model(inputs).predictions.numpy()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory + "/model")
            code = snippet("docs/source/faq.rst", "class ForecastService")
            code = code.replace("'./my_model'", repr(directory + "/model"))
            code = code.replace("'./saved_model/1'", repr(directory + "/saved_model/1"))
            exec(code, {})
            service = tf.saved_model.load(directory + "/saved_model/1")
            result = service.signatures["serving_default"](inputs=inputs)["predictions"]
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_documented_convolutional_models(self):
        for name in ("tcn", "wavenet"):
            with self.subTest(model=name):
                namespace = {}
                exec(snippet("docs/source/models.rst", "for_model('" + name + "')"), namespace)
                predictions = namespace["model"](tf.ones([2, 24, 1])).predictions
                self.assertEqual(predictions.shape, (2, 24, 1))
                config = namespace["config"]
                self.assertEqual(len(config.kernel_sizes), len(config.dilation_rates))

    def test_spatial_example(self):
        from examples.run_spatial import main

        main()
