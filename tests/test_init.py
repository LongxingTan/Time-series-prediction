import importlib
import unittest

import tfts


class PackageInitTest(unittest.TestCase):
    def test_public_exports(self):
        tfts = importlib.reload(__import__("tfts"))
        self.assertEqual(tfts.__version__, "0.0.5")
        self.assertIn("Trainer", tfts.__all__)
        self.assertNotIn("BenchmarkRunner", tfts.__all__)
        self.assertIn("FramePipeline", tfts.__all__)
        self.assertIn("TemporalSplit", tfts.__all__)
        self.assertIn("Estimator", tfts.__all__)
        self.assertIn("Forecaster", tfts.__all__)

    def test_unknown_attribute_raises_attribute_error(self):
        with self.assertRaisesRegex(AttributeError, "missing_attribute"):
            _ = tfts.missing_attribute


if __name__ == "__main__":
    unittest.main()
