"""Dataset adapter for experiment tensors stored in NumPy archives."""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from benchmark.base import Dataset


class NpzDataset(Dataset):
    """Load train/validation arrays without writing a Python dataset class."""

    name = "npz"
    description = "Generic train/validation NPZ archives."

    def prepare_data(self, **kwargs):
        return self._load(kwargs["train_path"], kwargs)

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[object, np.ndarray], Tuple[object, np.ndarray]]:
        try:
            train_path = kwargs["train_path"]
            valid_path = kwargs["valid_path"]
        except KeyError as exc:
            raise ValueError("npz requires train_path and valid_path") from exc
        return self._load(train_path, kwargs), self._load(valid_path, kwargs)

    @staticmethod
    def _load(path: Union[str, Path], config: Dict[str, object]):
        input_keys = config.get("input_keys")
        target_key = config.get("target_key", "y")
        if not isinstance(input_keys, list) or not input_keys:
            raise ValueError("npz input_keys must be a non-empty list")
        with np.load(path) as archive:
            missing = set(input_keys + [target_key]) - set(archive.files)
            if missing:
                raise ValueError(f"Missing arrays in {path}: {sorted(missing)}")
            if len(input_keys) == 1:
                inputs = archive[input_keys[0]].copy()
            else:
                inputs = {key: archive[key].copy() for key in input_keys}
            target = archive[target_key].copy()
        return inputs, target
