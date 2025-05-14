"""Tests for the feature registry module."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from tfts.features.registry import FeatureRegistry, feature_registry, registry


class TestFeatureRegistry(unittest.TestCase):
    """Test cases for the FeatureRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = FeatureRegistry()

    def test_init(self):
        """Test initialization of FeatureRegistry."""
        self.assertEqual(self.registry.columns, [])
        self.assertIsInstance(self.registry.columns, list)

    def test_register_single_feature(self):
        """Test registering a single feature."""
        self.registry.register("feature1")
        self.assertEqual(self.registry.columns, ["feature1"])

    def test_register_multiple_features(self):
        """Test registering multiple features."""
        features = ["feature1", "feature2", "feature3"]
        self.registry.register(features)
        self.assertEqual(self.registry.columns, features)

    def test_register_invalid_type(self):
        """Test registering invalid feature types."""
        with self.assertRaises(TypeError):
            self.registry.register(123)
        with self.assertRaises(TypeError):
            self.registry.register(["feature1", 123])

    def test_register_invalid_chars(self):
        """Test registering features with invalid characters."""
        with self.assertRaises(ValueError):
            self.registry.register("feature/1")
        with self.assertRaises(ValueError):
            self.registry.register("feature*1")

    def test_register_empty_string(self):
        """Test registering empty feature names."""
        with self.assertRaises(ValueError):
            self.registry.register("")
        with self.assertRaises(ValueError):
            self.registry.register(["feature1", ""])

    def test_get_features(self):
        """Test getting features."""
        features = ["feature1", "feature2"]
        self.registry.register(features)
        returned_features = self.registry.get_features()
        self.assertEqual(returned_features, features)
        # Test that returned list is a copy
        returned_features.append("feature3")
        self.assertEqual(self.registry.columns, features)

    def test_hash_features(self):
        """Test feature hashing."""
        features = ["feature1", "feature2", "feature3"]
        self.registry.register(features)
        hash1 = self.registry.hash_features()

        # Test that hash is consistent
        self.registry = FeatureRegistry()
        self.registry.register(features)
        hash2 = self.registry.hash_features()
        self.assertEqual(hash1, hash2)

        # Test that hash changes with different features
        self.registry = FeatureRegistry()
        self.registry.register(["feature1", "feature2"])
        hash3 = self.registry.hash_features()
        self.assertNotEqual(hash1, hash3)

    def test_save_and_load(self):
        """Test saving and loading features."""
        features = ["feature1", "feature2", "feature3"]
        self.registry.register(features)

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "features.json")

            # Test saving
            self.assertTrue(self.registry.save(filepath))
            self.assertTrue(os.path.exists(filepath))

            # Verify saved content
            with open(filepath, "r") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["features"], features)

            # Test loading
            new_registry = FeatureRegistry()
            self.assertTrue(new_registry.load(filepath))
            self.assertEqual(new_registry.columns, features)

    def test_save_invalid_path(self):
        """Test saving to invalid path."""
        with self.assertRaises(Exception):
            self.registry.save("/invalid/path/features.json")

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            self.registry.load("nonexistent.json")

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            f.write("invalid json")
            f.flush()
            with self.assertRaises(json.JSONDecodeError):
                self.registry.load(f.name)

    def test_load_invalid_format(self):
        """Test loading file with invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump({"invalid": "format"}, f)
            f.flush()
            with self.assertRaises(ValueError):
                self.registry.load(f.name)

    def test_repr(self):
        """Test string representation."""
        self.assertEqual(str(self.registry), "FeatureRegistry with 0 features")
        self.registry.register("feature1")
        self.assertEqual(str(self.registry), "FeatureRegistry with 1 features")


class TestRegistryDecorator(unittest.TestCase):
    """Test cases for the registry decorator."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear the global registry before each test
        feature_registry.columns = []

    def test_registry_decorator(self):
        """Test the registry decorator."""

        @registry
        def get_features():
            return ["feature1", "feature2"]

        result = get_features()
        self.assertEqual(result, ["feature1", "feature2"])
        self.assertEqual(feature_registry.columns, ["feature1", "feature2"])

    def test_registry_decorator_single_feature(self):
        """Test the registry decorator with single feature."""

        @registry
        def get_feature():
            return "feature1"

        result = get_feature()
        self.assertEqual(result, "feature1")
        self.assertEqual(feature_registry.columns, ["feature1"])


if __name__ == "__main__":
    unittest.main()
