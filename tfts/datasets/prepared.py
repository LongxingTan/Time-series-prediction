"""Small prepared dataset recipes for examples and model smoke tests.

For real experiments use the raw load_* functions and an explicit FramePipeline.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .air_passengers import load_air_passengers
from .electricity import load_electricity
from .synthetic import load_ar, load_sine
from .traffic import load_traffic
from .volatility import load_volatility


def get_data(
    name: str = "sine", train_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.1, **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    if name == "sine":
        return get_sine(train_length, predict_sequence_length, test_size=test_size, **kwargs)

    elif name == "airpassengers":
        return get_air_passengers(train_length, predict_sequence_length, test_size=test_size)

    elif name == "ar":
        return load_ar(**kwargs)

    elif name == "volatility":
        return load_volatility()

    elif name == "electricity":
        return load_electricity()

    elif name == "traffic":
        return load_traffic()

    else:
        raise ValueError(
            f"unsupported data of {name} yet, try 'sine', 'airpassengers', 'ar', 'volatility', 'electricity', 'traffic'"
        )


def get_sine(train_sequence_length=24, predict_sequence_length=8, test_size=0.2, n_examples=100, seed=None):
    if min(train_sequence_length, predict_sequence_length, n_examples) < 1:
        raise ValueError("sequence lengths and n_examples must be positive")
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    frame = load_sine(n_examples, train_sequence_length + predict_sequence_length, seed)
    values = frame.value.to_numpy().reshape(n_examples, -1, 1)
    x, y = values[:, :train_sequence_length], values[:, train_sequence_length:]
    if not test_size:
        return x, y
    cutoff = int(n_examples * (1 - test_size))
    return (x[:cutoff], y[:cutoff]), (x[cutoff:], y[cutoff:])


def get_air_passengers(train_sequence_length=24, predict_sequence_length=8, test_size=0.2):
    from tfts.data.pipeline import FramePipeline
    from tfts.data.schema import Schema

    pipeline = FramePipeline(
        Schema("date", ("value",)), train_sequence_length, predict_sequence_length, normalize="minmax", freq="MS"
    )
    prepared, train, valid = pipeline.fit(load_air_passengers(), validation=test_size)

    def arrays(windows):
        inputs, targets = next(iter(pipeline.dataset(prepared, windows, batch_size=len(windows))))
        return inputs["past_values"].numpy(), targets.numpy()

    return arrays(train) if valid is None else (arrays(train), arrays(valid))
