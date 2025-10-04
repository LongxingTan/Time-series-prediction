"""tfts Generator"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf


class GenerationConfig:
    def __init__(self, **kwargs) -> None:
        self.max_length = kwargs.pop("max_length", 20)


class GenerationMixin:
    """
    A class containing auto-regressive generation, to be used as a mixin.
    """

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return

    def generate(
        self,
        inputs: Union[pd.DataFrame, np.ndarray],
        generation_config: Dict[str, Any] = None,
        logits_processor=None,
        seed=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate time series predictions in an auto-regressive manner.

        Args:
            inputs: Initial input sequence as DataFrame or numpy array
            generation_config: Configuration for generation with the following keys:
                - steps: Number of steps to predict
                - time_idx: Name of the time index column
                - time_step: Time step increment for prediction
                - group_columns: List of group identifier columns
                - add_features_func: Function to add features after each prediction

        Returns:
            DataFrame with original inputs and generated predictions
        """
        generation_config = generation_config or {}
        steps = generation_config.get("steps", 1)
        time_idx = generation_config.get("time_idx", self.time_idx)
        time_step = generation_config.get("time_step", 1)
        group_columns = generation_config.get("group_columns", self.group_column)
        add_features_func = generation_config.get("add_features_func", None)

        # Convert inputs to DataFrame if needed
        if isinstance(inputs, np.ndarray):
            features = self.get_feature_names()
            if len(features) != inputs.shape[1]:
                raise ValueError(f"Input array shape {inputs.shape} doesn't match feature count {len(features)}")
            inputs_df = pd.DataFrame(inputs, columns=features)

            # Add time index if not present
            if time_idx not in inputs_df.columns:
                # Create a time index
                max_time = inputs_df[time_idx].max() if time_idx in inputs_df.columns else 0
                inputs_df[time_idx] = np.arange(max_time + 1, max_time + len(inputs_df) + 1)
        else:
            inputs_df = inputs.copy()

        # Ensure inputs are sorted by time index
        inputs_df = inputs_df.sort_values(by=time_idx)

        # Create a copy to store results
        results_df = inputs_df.copy()
        last_time_idx = results_df[time_idx].max()

        # Get the sequence length for the model
        seq_length = self.train_sequence_length

        # Predict one step at a time and add to results
        for step in range(steps):
            # Get the latest sequence
            if len(results_df) < seq_length:
                # Not enough data yet, pad with zeros
                pad_length = seq_length - len(results_df)
                feature_cols = [col for col in results_df.columns if col != time_idx]
                pad_df = pd.DataFrame(0, index=range(pad_length), columns=feature_cols)
                pad_df[time_idx] = np.arange(
                    results_df[time_idx].min() - pad_length * time_step, results_df[time_idx].min(), time_step
                )
                temp_df = pd.concat([pad_df, results_df], ignore_index=True)
            else:
                temp_df = results_df.tail(seq_length)

            # Extract features for prediction
            input_features = temp_df[[col for col in temp_df.columns if col != time_idx]].values
            input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

            # Make prediction
            prediction = self.predict(input_features)

            # Create a new row for the prediction
            new_time = last_time_idx + (step + 1) * time_step
            new_row = {time_idx: new_time}

            # Add group identifiers if they exist
            if group_columns:
                for col in group_columns:
                    new_row[col] = results_df[col].iloc[-1]

            # Add prediction for target columns
            for i, target in enumerate(self.target):
                new_row[target] = prediction[0, 0, i]  # Assuming [batch, time, feature] format

            # Add new row to results
            new_df = pd.DataFrame([new_row])

            # Apply custom feature generation if provided
            if add_features_func:
                new_df = add_features_func(new_df, results_df)

            # Append to results
            results_df = pd.concat([results_df, new_df], ignore_index=True)

        return results_df

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make a prediction using the model.

        This method should be implemented by the child class.

        Args:
            inputs: Input features of shape [batch, time, features]

        Returns:
            Predictions of shape [batch, prediction_length, targets]
        """
        raise NotImplementedError("Subclasses must implement predict method")
