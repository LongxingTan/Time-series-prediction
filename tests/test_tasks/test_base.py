from dataclasses import FrozenInstanceError, dataclass
from typing import Optional
import unittest

import tensorflow as tf

from tfts.tasks.base import BaseTask, ModelOutput


class TestBaseTask(unittest.TestCase):
    def test_base_task_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseTask()


class TestModelOutput(unittest.TestCase):
    def setUp(self):
        @dataclass(frozen=True)
        class TestOutput(ModelOutput):
            value1: Optional[tf.Tensor] = None
            value2: Optional[tf.Tensor] = None
            value3: Optional[int] = None

        self.TestOutput = TestOutput

    def test_to_dict_excludes_none_values(self):
        output = self.TestOutput(value1=tf.constant([1, 2, 3]), value3=42)
        self.assertEqual(set(output.to_dict()), {"value1", "value3"})

    def test_replace_preserves_original(self):
        original = self.TestOutput(value1=tf.constant([1]))
        updated = original.replace(value2=tf.constant([2]))
        self.assertIsNone(original.value2)
        self.assertEqual(int(updated.value2[0]), 2)

    def test_output_is_immutable_and_not_indexable(self):
        output = self.TestOutput(value1=tf.constant([1]))
        with self.assertRaises(FrozenInstanceError):
            output.value3 = 3
        with self.assertRaises(TypeError):
            _ = output[0]
