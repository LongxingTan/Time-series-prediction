"""Dataset cache and download configuration."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from tfts.constants import TFTS_DATASETS_CACHE

TS_DATASETS_URL = {
    "air_passengers": {
        "url": "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",  # noqa: E501
        "format": "csv",
        "freq": "MS",
    },
    "volatility": {
        "url": "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip",
        "format": "zip",
        "csv_inside": "oxfordmanrealizedvolatilityindices.csv",
    },
    "electricity": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
        "format": "zip",
        "freq": "15T",
        "csv_inside": "LD2011_2014.txt",
    },
    "traffic": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip",
        "format": "zip",
        "freq": "H",
        "csv_inside": "PEMS_train",
    },
    "favorita": {
        "url": "https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data",
        "format": "kaggle",
    },
    "m5": {
        "url": "https://www.kaggle.com/c/m5-forecasting-accuracy/data",
        "format": "kaggle",
    },
}


def download_and_extract(name: str) -> str:
    """Robust download utility using Keras get_file logic."""
    from tensorflow.keras.utils import get_file

    if name not in TS_DATASETS_URL:
        raise ValueError(f"Dataset {name} configuration not found.")

    config = TS_DATASETS_URL[name]
    cache_dir = os.path.join(TFTS_DATASETS_CACHE, name)
    os.makedirs(cache_dir, exist_ok=True)

    path = get_file(
        fname=name, origin=config["url"], cache_subdir=cache_dir, extract=(config.get("format", "zip") == "zip")
    )
    return path
