"""Load volatility observations."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ._download import TS_DATASETS_URL, download_and_extract


def load_volatility() -> pd.DataFrame:
    data_dir = download_and_extract("volatility")
    csv_path = os.path.join(data_dir, TS_DATASETS_URL["volatility"]["csv_inside"])

    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime([str(s).split("+")[0] for s in df.index])
    df = df.reset_index().rename(columns={"index": "date"})

    return df
