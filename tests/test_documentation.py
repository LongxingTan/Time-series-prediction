"""Execute published quickstarts directly so API changes cannot leave stale copies."""

import os
from pathlib import Path
import re
import tempfile
import textwrap
import unittest

import matplotlib
import tensorflow as tf

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTest(unittest.TestCase):
    def execute_blocks(self, path, blocks):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                namespace = {"__name__": "__documentation__"}
                for index, source in enumerate(blocks):
                    with self.subTest(document=path, block=index + 1):
                        exec(compile(textwrap.dedent(source), str(ROOT / path), "exec"), namespace)
                if "build_model" in namespace:
                    model = namespace["build_model"]()
                    result = model(tf.ones([2, namespace["train_length"], namespace["num_train_features"]]))
                    self.assertEqual(result.shape, (2, namespace["predict_sequence_length"], 1))
            finally:
                os.chdir(previous)
                tf.keras.backend.clear_session()

    def test_readme_quickstarts(self):
        for path in ("README.md", "README_CN.md"):
            source = (ROOT / path).read_text()
            blocks = re.findall(r"```python\s*\n(.*?)```", source, re.DOTALL)
            self.assertTrue(blocks)
            self.execute_blocks(path, blocks)

    def test_architecture_examples(self):
        path = "docs/source/architecture.rst"
        source = (ROOT / path).read_text()
        blocks = re.findall(r"\.\. code-block:: python\n\n((?:(?:   [^\n]*|)\n)+)", source)
        self.assertEqual(len(blocks), 2)
        self.execute_blocks(path, blocks)
