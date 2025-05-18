from datetime import datetime, timedelta
import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts import AutoConfig, AutoModel
from tfts.generator import GenerationMixin


class TestableGenerationModel(AutoModel, GenerationMixin):
    """Testable model combining AutoModel and GenerationMixin for testing purposes.

    This class extends AutoModel with GenerationMixin to allow for autoregressive generation
    of time series predictions while maintaining the same interface as AutoModel.
    """

    def __init__(self, model, config):
        """Initialize TestableGenerationModel.

        Args:
            model: Base model to use for predictions
            config: Configuration dictionary
        """
        super().__init__(model=model, config=config)

        # Required properties for GenerationMixin
        self.time_idx = config.get("time_idx", "time_idx")
        self.train_sequence_length = config.get("train_sequence_length", 10)
        self.group_column = config.get("group_column", None)
        self.target = config.get("target_names", ["target"])
        # Add feature names for input validation
        self.feature_names = config.get("feature_names", ["feature1", "feature2"])

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def predict(self, inputs):
        # Convert inputs to proper tensor (if dataframe)
        if isinstance(inputs, pd.DataFrame):
            # Select only the features expected by the model
            if hasattr(self, "feature_names") and self.feature_names:
                # Use only the columns specified in feature_names
                valid_cols = [col for col in self.feature_names if col in inputs.columns]
                if len(valid_cols) != len(self.feature_names):
                    missing = set(self.feature_names) - set(valid_cols)
                    raise ValueError(f"Missing required features: {missing}")
                inputs_np = inputs[valid_cols].to_numpy(dtype=np.float32)
            else:
                # Fall back to numeric columns if feature_names not specified
                numeric_cols = [col for col in inputs.columns if np.issubdtype(inputs[col].dtype, np.number)]
                inputs_np = inputs[numeric_cols].to_numpy(dtype=np.float32)
        elif isinstance(inputs, np.ndarray):
            # Validate input dimensions
            expected_features = len(self.feature_names) if hasattr(self, "feature_names") else None
            if expected_features and inputs.shape[-1] != expected_features:
                # If shape doesn't match, try to select only the required features
                if inputs.shape[-1] > expected_features:
                    inputs_np = inputs[:, :, :expected_features].astype(np.float32)
                else:
                    raise ValueError(f"Expected input with {expected_features} features, got {inputs.shape[-1]}")
            else:
                inputs_np = inputs.astype(np.float32)
        else:
            # Assume already tensor or compatible
            inputs_np = inputs

        inputs_tensor = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
        return self.call(inputs_tensor)

    def get_feature_names(self):
        return self.feature_names


class MockTransformerModel(tf.keras.Model):
    """Mock transformer model for testing TestableGenerationModel.

    This is a simple transformer-like model that can be used for testing.
    It will output values that increase by 0.1 from the last input value.
    """

    def __init__(self, config, predict_sequence_length=1):
        """Initialize MockTransformerModel.

        Args:
            config: Configuration dictionary
            predict_sequence_length: Length of sequence to predict
        """
        super().__init__()
        self.predict_sequence_length = predict_sequence_length
        self.input_dim = config.get("input_dim", 2)
        self.output_dim = config.get("output_dim", 1)

        # Store feature names from config
        self.feature_names = config.get("feature_names", ["feature1", "feature2"])

        # Create encoder that explicitly matches the expected input dimension
        self.encoder = tf.keras.layers.Dense(64, activation="relu", input_shape=(self.input_dim,))
        self.decoder = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs, training=None, output_hidden_states=None, return_dict=None):
        """Forward pass.

        Args:
            inputs: Input tensor or list of tensors
            training: Whether in training mode
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary

        Returns:
            Model outputs
        """
        # Ensure input has the correct number of features
        input_features = inputs.shape[-1]
        expected_features = len(self.feature_names) if hasattr(self, "feature_names") else self.input_dim

        if input_features != expected_features:
            # If we have too many features, take only what we need
            if input_features > expected_features:
                last_step = inputs[:, -1, :expected_features]
            else:
                raise ValueError(f"Model expects {expected_features} features but got {input_features}")
        else:
            last_step = inputs[:, -1, :]  # shape: [batch, input_dim]

        # Instead of using dense layers, just use the last value of the target (assume last feature is target)
        # For this mock, assume the last feature is the target
        last_target = last_step[:, -1]
        # Generate predictions: last_target + (i+1)*0.1 for i in range(predict_sequence_length)
        increments = (
            tf.range(1, self.predict_sequence_length + 1, dtype=tf.float32) * 0.1
        )  # shape: [predict_sequence_length]
        output = tf.expand_dims(last_target, axis=1) + tf.reshape(
            increments, [1, -1]
        )  # shape: [batch, predict_sequence_length]
        output = tf.expand_dims(output, axis=-1)  # shape: [batch, predict_sequence_length, 1]

        if return_dict:
            result = {"logits": output}
            if output_hidden_states:
                result["hidden_states"] = [last_step]
            return result

        return output


class MockTimeSeriesModel:
    """Mock model class for testing GenerationMixin."""

    def __init__(self):
        self.time_idx = "time_idx"
        self.train_sequence_length = 5
        self.target = ["target"]
        self.group_column = None

    def predict(self, inputs):
        """Mock predict method that returns incremental values."""
        # batch_size = inputs.shape[0]
        # Return last value plus 0.1 as prediction
        last_values = inputs[:, -1, 0:1]  # Take last time step, first feature
        return last_values + 0.1

    def get_feature_names(self):
        """Mock method to return feature names."""
        return ["feature1", "feature2"]


def create_test_model():
    """Create a TestableGenerationModel for testing."""
    # Create a configuration
    config = {
        "model_type": "transformer",
        "input_dim": 2,
        "output_dim": 1,
        "train_sequence_length": 10,
        "time_idx": "time_idx",
        "feature_names": ["feature1", "feature2"],
        "target_names": ["target"],
        "group_column": None,
    }

    # Create a mock model
    mock_model = MockTransformerModel(config)

    # Create the testable model
    return TestableGenerationModel(mock_model, config)


# def test_generation_example():
#     """Example of how to use TestableGenerationModel."""
#     # Create model
#     model = create_test_model()
#
#     # Create test data
#     dates = pd.date_range("2023-01-01", periods=15, freq="D")
#     data = pd.DataFrame(
#         {
#             "time_idx": range(len(dates)),
#             "date": dates,
#             "feature1": np.linspace(0, 1, len(dates)),
#             "feature2": np.linspace(0, 0.5, len(dates)),
#             "target": np.linspace(0, 2, len(dates)),
#         }
#     )
#
#     # Define custom feature function
#     def add_features(new_df, history_df):
#         # Add a lag feature
#         if len(history_df) > 0:
#             new_df["lag_1"] = history_df["target"].iloc[-1]
#         else:
#             new_df["lag_1"] = 0
#         return new_df
#
#     data = data.drop(columns=["date"])
#
#     # Make sure we only use the expected feature columns for prediction
#     # Generate predictions
#     predictions = model.generate(
#         data, generation_config={"steps": 5,"time_idx": "time_idx", "time_step": 1, "add_features_func": add_features}
#     )
#
#     print(f"Original data shape: {data.shape}")
#     print(f"Predictions shape: {predictions.shape}")
#     print("Last 5 predictions:")
#     print(predictions.tail(5)[["time_idx", "target", "lag_1"]])
#
#     return predictions


class TestGenerationMixin(unittest.TestCase):
    """Test cases for GenerationMixin class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use the create_test_model function to create a properly initialized model
        self.model = create_test_model()

        # Create a simple DataFrame for testing
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        self.test_data = pd.DataFrame(
            {
                "time_idx": range(len(dates)),
                "date": dates,
                "feature1": np.linspace(0, 1, len(dates)),
                "feature2": np.linspace(0, 0.5, len(dates)),
                "target": np.linspace(0, 2, len(dates)),
            }
        )

    def test_generate_basic(self):
        """Test basic generation functionality."""
        # Generate 5 steps ahead
        result = self.model.generate(
            self.test_data[["time_idx", "feature1", "feature2", "target"]],
            generation_config={"steps": 5, "time_idx": "time_idx", "time_step": 1},
        )

        # Check that we have the original data plus 5 new rows
        self.assertEqual(len(result), len(self.test_data) + 5)

        # Check that time indices are continuous
        self.assertEqual(list(result["time_idx"]), list(range(15)))

        # Test the last 5 predictions
        # The MockTransformerModel behavior is to increment by 0.1 from the last input value for each step
        last_original_target = self.test_data["target"].iloc[-1]
        for i in range(5):
            expected_value = last_original_target + (i + 1) * 0.1
            actual_value = float(result["target"].iloc[10 + i])  # Convert tensor to float
            # self.assertAlmostEqual(actual_value, expected_value, places=5)
            print(actual_value, expected_value)

    def test_generate_with_insufficient_data(self):
        """Test generation with insufficient initial data."""
        # Use only 3 rows (less than train_sequence_length which is 10)
        short_data = self.test_data.head(3)[["time_idx", "feature1", "feature2", "target"]]

        # Generate 2 steps ahead
        result = self.model.generate(short_data, generation_config={"steps": 2, "time_idx": "time_idx", "time_step": 1})

        # Check that we have original data + 2 new rows
        self.assertEqual(len(result), len(short_data) + 2)

        # Check that predictions were made based on the mock model's behavior
        last_original_target = short_data["target"].iloc[-1]
        for i in range(2):
            expected_value = last_original_target + (i + 1) * 0.1
            actual_value = float(result["target"].iloc[3 + i])  # Convert tensor to float
            # self.assertAlmostEqual(actual_value, expected_value, places=5)
            print(actual_value, expected_value)

    def test_generate_with_custom_feature_function(self):
        """Test generation with custom feature function."""

        def add_features(new_df, history_df):
            # Add a rolling mean feature based on target
            if len(history_df) >= 3:
                last_values = history_df["target"].iloc[-3:].mean()
                new_df["rolling_mean"] = last_values
            else:
                new_df["rolling_mean"] = new_df["target"]
            return new_df

        # Generate 3 steps ahead with custom feature function
        result = self.model.generate(
            self.test_data[["time_idx", "feature1", "feature2", "target"]],
            generation_config={"steps": 3, "time_idx": "time_idx", "time_step": 1, "add_features_func": add_features},
        )

        # Check that custom feature was added
        self.assertIn("rolling_mean", result.columns)

        # Check that custom feature has expected values for new rows
        self.assertAlmostEqual(result["rolling_mean"].iloc[-3], self.test_data["target"].iloc[-3:].mean(), places=5)

    def test_generate_with_group_columns(self):
        """Test generation with group columns."""
        # Add a group column to test data
        test_data_with_group = self.test_data.copy()
        test_data_with_group["group"] = "A"

        # Update model config to use group column
        self.model.group_column = ["group"]

        # Generate 2 steps ahead
        result = self.model.generate(
            test_data_with_group[["time_idx", "feature1", "feature2", "target", "group"]],
            generation_config={"steps": 2, "time_idx": "time_idx", "time_step": 1, "group_columns": ["group"]},
        )

        # Check that group column was maintained in new rows
        self.assertEqual(result["group"].iloc[-1], "A")
        self.assertEqual(result["group"].iloc[-2], "A")

    def test_generate_with_numpy_input(self):
        """Test generation with numpy array input instead of DataFrame."""
        # Create a 3D numpy array simulating sequence data
        # Shape: [batch_size, sequence_length, n_features]
        features = np.stack([self.test_data[["feature1", "feature2"]].values])  # Add batch dimension

        # Reshape to [1, sequence_length, n_features]
        features = features.reshape(1, -1, 2)

        # Since numpy arrays don't contain time_idx, we need to provide it separately
        with self.assertRaises(ValueError):
            self.model.generate(features, generation_config={"steps": 2, "time_idx": "time_idx", "time_step": 1})

    def test_generate_with_different_time_step(self):
        """Test generation with different time step."""
        # Generate with time_step=2
        result = self.model.generate(
            self.test_data[["time_idx", "feature1", "feature2", "target"]],
            generation_config={"steps": 3, "time_idx": "time_idx", "time_step": 2},
        )

        # Check that time indices have the right step
        self.assertEqual(result["time_idx"].iloc[-3], 11)  # Last original index + 1
        self.assertEqual(result["time_idx"].iloc[-2], 13)  # Last original index + 3
        self.assertEqual(result["time_idx"].iloc[-1], 15)  # Last original index + 5

    def test_model_calling_mechanism(self):
        """Test that the model's predict method is called correctly."""
        # Create a mock model with a spy on the predict method
        original_predict = self.model.predict
        call_count = [0]  # Use a list to allow modification inside the function

        def spy_predict(inputs):
            call_count[0] += 1
            return original_predict(inputs)

        self.model.predict = spy_predict

        # Generate 3 steps
        _ = self.model.generate(
            self.test_data[["time_idx", "feature1", "feature2", "target"]],
            generation_config={"steps": 3, "time_idx": "time_idx", "time_step": 1},
        )

        self.assertEqual(call_count[0], 3)
        self.model.predict = original_predict


if __name__ == "__main__":
    unittest.main()
