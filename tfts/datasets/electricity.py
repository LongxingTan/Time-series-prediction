"""Load electricity observations."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ._download import TS_DATASETS_URL, download_and_extract


def load_electricity() -> pd.DataFrame:
    data_dir = download_and_extract("electricity")
    csv_path = os.path.join(data_dir, TS_DATASETS_URL["electricity"]["csv_inside"])

    # Industrial datasets are often large; use specific separators
    df = pd.read_csv(csv_path, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.resample("1H").mean().replace(0.0, np.nan)

    # Melt to long format (productive for TimeSeriesSequence)
    df = df.reset_index().melt(id_vars="index", var_name="id", value_name="power_usage")
    df = df.rename(columns={"index": "date"}).dropna()
    return df
