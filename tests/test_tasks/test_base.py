from dataclasses import dataclass
from typing import Optional, Tuple
import unittest

import tensorflow as tf

from tfts.tasks.base import BaseTask, ModelOutput


class TestBaseTask(unittest.TestCase):
    """Test BaseTask abstract class"""

    def test_base_task_is_abstract(self):
        """Verify BaseTask cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseTask()


class TestModelOutput(unittest.TestCase):
    """Test ModelOutput base class"""

    def setUp(self):
        @dataclass
        class TestOutput(ModelOutput):
            value1: tf.Tensor = None
            value2: Optional[tf.Tensor] = None
            value3: int = None

        self.TestOutput = TestOutput

    def test_post_init_populates_dict(self):
        """Test that __post_init__ correctly populates the OrderedDict"""
        tensor1 = tf.constant([1, 2, 3])
        tensor2 = tf.constant([4, 5, 6])
        output = self.TestOutput(value1=tensor1, value2=tensor2, value3=42)

        self.assertIn("value1", output)
        self.assertIn("value2", output)
        self.assertIn("value3", output)
        self.assertEqual(len(output), 3)

    def test_post_init_excludes_none_values(self):
        """Test that None values are not added to the dict"""
        tensor1 = tf.constant([1, 2, 3])
        output = self.TestOutput(value1=tensor1)

        self.assertIn("value1", output)
        self.assertNotIn("value2", output)
        self.assertNotIn("value3", output)
        self.assertEqual(len(output), 1)

    def test_getitem_with_int(self):
        """Test indexing with integer returns tuple element"""
        tensor1 = tf.constant([1, 2, 3])
        tensor2 = tf.constant([4, 5, 6])
        output = self.TestOutput(value1=tensor1, value2=tensor2)

        self.assertTrue(tf.reduce_all(output[0] == tensor1))
        self.assertTrue(tf.reduce_all(output[1] == tensor2))

    def test_getitem_with_string(self):
        """Test indexing with string returns dict value"""
        tensor1 = tf.constant([1, 2, 3])
        output = self.TestOutput(value1=tensor1)

        self.assertTrue(tf.reduce_all(output["value1"] == tensor1))

    def test_to_tuple(self):
        """Test to_tuple returns only non-None values"""
        tensor1 = tf.constant([1, 2, 3])
        tensor2 = tf.constant([4, 5, 6])
        output = self.TestOutput(value1=tensor1, value2=tensor2)

        result = output.to_tuple()
        self.assertEqual(len(result), 2)
        self.assertTrue(tf.reduce_all(result[0] == tensor1))
        self.assertTrue(tf.reduce_all(result[1] == tensor2))

    def test_to_tuple_empty(self):
        """Test to_tuple with all None values"""
        output = self.TestOutput()
        result = output.to_tuple()
        self.assertEqual(len(result), 0)
