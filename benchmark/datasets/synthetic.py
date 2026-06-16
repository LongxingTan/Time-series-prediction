"""Synthetic benchmark datasets."""

import random
from typing import Any, Dict, Tuple

import numpy as np

from benchmark.base import Dataset
from benchmark.datasets.base import split_train_valid
from tfts.data.get_data import get_air_passengers, get_sine


class SineDataset(Dataset):
    """Synthetic sine wave benchmark dataset."""

    name = "sine"
    description = "Synthetic sine wave data."
    train_length = 24
    predict_sequence_length = 8
    num_features = 1

    def prepare_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        x, y = get_sine(
            train_sequence_length=self.train_length,
            predict_sequence_length=self.predict_sequence_length,
            test_size=0.0,
            n_examples=kwargs.get("n_examples", 100),
        )
        return x, y

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        train_len = kwargs.get("train_length") or self.train_length
        pred_len = kwargs.get("predict_sequence_length") or self.predict_sequence_length
        (x_train, y_train), (x_valid, y_valid) = get_sine(
            train_sequence_length=train_len,
            predict_sequence_length=pred_len,
            test_size=kwargs.get("test_size", 0.2),
            n_examples=kwargs.get("n_examples", 100),
        )
        return (x_train, y_train), (x_valid, y_valid)


class AirPassengersDataset(Dataset):
    """Air passengers benchmark dataset."""

    name = "air_passengers"
    description = "Airline passenger counts (1949-1960)."
    train_length = 24
    predict_sequence_length = 8
    num_features = 1

    def prepare_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        x, y = get_air_passengers(
            train_sequence_length=self.train_length,
            predict_sequence_length=self.predict_sequence_length,
            test_size=0.0,
        )
        return x, y

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        train_len = kwargs.get("train_length") or self.train_length
        pred_len = kwargs.get("predict_sequence_length") or self.predict_sequence_length
        (x_train, y_train), (x_valid, y_valid) = get_air_passengers(
            train_sequence_length=train_len,
            predict_sequence_length=pred_len,
            test_size=kwargs.get("test_size", 0.2),
        )
        return (x_train, y_train), (x_valid, y_valid)
