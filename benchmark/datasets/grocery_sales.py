"""Kaggle Favorita grocery sales dataset for TFTS benchmark.

Note:
    This is a concrete example of how to add a real-world Kaggle dataset.
    Users need to download the data from Kaggle and update ``data_path``.
"""

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from benchmark.base import Dataset
from benchmark.datasets.base import split_train_valid

logger = logging.getLogger(__name__)


class GrocerysalesDataset(Dataset):
    """Kaggle Favorita Grocery Sales dataset.

    Expected columns after preprocessing:
        - ``date`` (datetime)
        - ``store_nbr`` (int, group column)
        - ``item_nbr`` (int, group column)
        - ``unit_sales`` (float, target)
        - additional feature columns (e.g., ``onpromotion``)

    Example:
        >>> dataset = GrocerysalesDataset(data_path="/path/to/favorita")
        >>> (x_train, y_train), (x_valid, y_valid) = dataset.get_train_valid_split()
    """

    name = "grocery_sales"
    description = "Kaggle Favorita Grocery Sales Forecasting"
    train_length = 28
    predict_sequence_length = 14
    num_features = 1
    is_grouped = True
    target_column = "unit_sales"
    time_column = "date"
    group_column = ["store_nbr", "item_nbr"]

    def __init__(self, data_path: str = "", **kwargs):
        self.data_path = data_path or os.environ.get("FAVORITA_PATH", "")
        super().__init__()

    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess the raw Favorita data.

        Returns:
            Cleaned DataFrame ready for sequence generation.
        """
        if not self.data_path:
            logger.warning("No data_path provided for GrocerysalesDataset. " "Returning synthetic placeholder data.")
            return self._synthetic_placeholder()

        # --- Stub for actual data loading ---
        # Users should replace this with their actual preprocessing pipeline.
        # Example:
        # df = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        # ... preprocess ...
        # return df
        return self._synthetic_placeholder()

    def _synthetic_placeholder(self) -> pd.DataFrame:
        """Generate a small synthetic placeholder when real data is missing."""
        logger.warning("Using synthetic placeholder for GrocerysalesDataset")
        np.random.seed(42)
        n_stores = 5
        n_items = 3
        days = 365
        records = []
        for store in range(n_stores):
            for item in range(n_items):
                trend = np.linspace(10, 20, days) + np.random.normal(0, 2, days)
                seasonal = 5 * np.sin(2 * np.pi * np.arange(days) / 7)
                for day, val in enumerate(trend + seasonal):
                    records.append(
                        {
                            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                            "store_nbr": store,
                            "item_nbr": item,
                            "unit_sales": val,
                            "onpromotion": 0,
                        }
                    )
        return pd.DataFrame(records)

    def _generate_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sliding-window sequences from a DataFrame.

        Returns:
            x, y arrays where x.shape == (samples, train_length, 1),
            y.shape == (samples, predict_sequence_length, 1).
        """
        x_list, y_list = [], []
        for (store, item), group in df.groupby(["store_nbr", "item_nbr"]):
            group = group.sort_values("date")
            values = group["unit_sales"].values.astype(np.float32)
            n = len(values)
            train_len = self.train_length
            pred_len = self.predict_sequence_length
            for i in range(n - train_len - pred_len + 1):
                x_list.append(values[i : i + train_len].reshape(-1, 1))
                y_list.append(values[i + train_len : i + train_len + pred_len].reshape(-1, 1))
        return np.array(x_list), np.array(y_list)

    def prepare_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        df = self._load_and_preprocess()
        return self._generate_sequences(df)

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x, y = self.prepare_data(**kwargs)
        return split_train_valid(x, y, test_size=kwargs.get("test_size", 0.2))
