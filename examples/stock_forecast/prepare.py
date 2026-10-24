"""Optional yfinance example data."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_stock_data(ticker: str = "NVDA", start_date="2023-09-01", end_date="2024-03-15") -> pd.DataFrame:
    """
    Retrieve historical stock data for a given ticker symbol.
    """
    # Download data
    import yfinance as yf

    try:
        logger.info(f"Retrieving data for {ticker} from {start_date} to {end_date}")

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"No data returned for ticker {ticker}")
            raise ValueError(f"No data available for ticker: {ticker}")

        logger.info(f"Successfully retrieved {len(data)} records for {ticker}")
        return data

    except Exception as e:
        logger.exception("Stock data retrieval failed.")
        raise RuntimeError(f"Failed to fetch stock data: {e}")
