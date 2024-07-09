import unittest

import numpy as np
import tensorflow as tf

from tfts.models.auto_config import AutoConfig


class TestAutoModel(unittest.TestCase):
    def test_auto_config(self):
        config = AutoConfig.for_model("bert")
        print(config)
