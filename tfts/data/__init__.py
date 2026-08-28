"""tfts data"""

from .auto_preprocessor import AutoPreprocessor
from .get_data import get_air_passengers, get_data, get_sine
from .materializers import SequenceMaterializer, TabularBatch, TabularMaterializer
from .processor import DataProcessor
from .timeseries import TimeSeriesSequence
from .windowing import Window, WindowIndex, WindowIndexer, WindowSpec

__all__ = [
    "AutoPreprocessor",
    "DataProcessor",
    "SequenceMaterializer",
    "TabularBatch",
    "TabularMaterializer",
    "TimeSeriesSequence",
    "Window",
    "WindowIndex",
    "WindowIndexer",
    "WindowSpec",
    "get_air_passengers",
    "get_data",
    "get_sine",
]
