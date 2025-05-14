"""Feature registry for managing time series features.

This module provides a registry system for tracking and managing features used in time series
prediction models. It allows for feature registration, retrieval, and persistence.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """A registry for managing time series features.

    This class provides functionality to register, track, and persist features used in
    time series prediction models. It maintains a list of feature columns and provides
    methods to manipulate and query them.

    Attributes:
        columns (List[str]): List of registered feature column names.
    """

    def __init__(self) -> None:
        """Initialize an empty feature registry."""
        self.columns: List[str] = []
        logger.debug("Initialized empty feature registry")

    def register(self, cols: Union[str, List[str]]) -> None:
        """Register one or more feature columns.

        Args:
            cols: A single feature column name or a list of feature column names to register.

        Raises:
            TypeError: If cols is not a string or list of strings.
            ValueError: If any column name is empty or contains invalid characters.
        """
        if not isinstance(cols, (str, list)):
            raise TypeError("cols must be a string or list of strings")

        if isinstance(cols, str):
            cols = [cols]

        # Validate column names
        for col in cols:
            if not isinstance(col, str):
                raise TypeError(f"Column name must be string, got {type(col)}")
            if not col.strip():
                raise ValueError("Column name cannot be empty")
            if any(c in col for c in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
                raise ValueError(f"Column name contains invalid characters: {col}")

        self.columns.extend(cols)
        logger.debug(f"Registered {len(cols)} features: {cols}")

    def get_features(self) -> List[str]:
        """Get all registered feature columns.

        Returns:
            List[str]: List of all registered feature column names.
        """
        return self.columns.copy()  # Return a copy to prevent external modification

    def hash_features(self) -> str:
        """Generate a hash of the registered features.

        This method creates an MD5 hash of the sorted feature names, which can be used
        for versioning or caching purposes.

        Returns:
            str: MD5 hash of the sorted feature names.
        """
        return hashlib.md5(",".join(sorted(self.columns)).encode()).hexdigest()

    def save(self, filepath: str = "feature_columns.json") -> bool:
        """Save the registered features to a JSON file.

        Args:
            filepath: Path where the features should be saved.

        Returns:
            bool: True if save was successful.

        Raises:
            IOError: If the file cannot be written.
            ValueError: If the filepath is invalid.
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as file:
                json.dump({"features": self.columns}, file, indent=2)
            logger.info(f"Saved {len(self.columns)} features to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save features to {filepath}: {str(e)}")
            raise

    def load(self, filepath: str = "feature_columns.json") -> bool:
        """Load features from a JSON file.

        Args:
            filepath: Path to the JSON file containing features.

        Returns:
            bool: True if load was successful.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If the file format is invalid.
        """
        try:
            with open(filepath, "r") as file:
                data = json.load(file)

            if not isinstance(data, dict) or "features" not in data:
                raise ValueError("Invalid feature file format")

            if not isinstance(data["features"], list):
                raise ValueError("Features must be a list")

            self.columns = data["features"]
            logger.info(f"Loaded {len(self.columns)} features from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load features from {filepath}: {str(e)}")
            raise

    def __repr__(self) -> str:
        """String representation of the feature registry.

        Returns:
            str: A string describing the registry and its feature count.
        """
        return f"FeatureRegistry with {len(self.columns)} features"


# Global feature registry instance
feature_registry = FeatureRegistry()


def registry(func: Callable) -> Callable:
    """Decorator to register features returned by a function.

    This decorator automatically registers any features returned by the decorated
    function with the global feature registry.

    Args:
        func: The function to decorate.

    Returns:
        Callable: The decorated function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            # Register only the new columns that were added
            original_cols = set(args[0].columns)  # args[0] is the input DataFrame
            new_cols = [col for col in result.columns if col not in original_cols]
            feature_registry.register(new_cols)
        else:
            feature_registry.register(result)
        return result

    return wrapper
