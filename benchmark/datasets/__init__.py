"""Built-in datasets for the TFTS benchmark system."""

from benchmark.datasets.grocery_sales import GrocerysalesDataset
from benchmark.datasets.legacy_examples import CMIDetectSleepStatesDataset, ForecastingStickerSalesDataset
from benchmark.datasets.m4 import M4Dataset
from benchmark.datasets.npz import NpzDataset
from benchmark.datasets.recruit_restaurant import RecruitRestaurantDataset
from benchmark.datasets.synthetic import AirPassengersDataset, SineDataset

__all__ = [
    "SineDataset",
    "AirPassengersDataset",
    "GrocerysalesDataset",
    "RecruitRestaurantDataset",
    "ForecastingStickerSalesDataset",
    "CMIDetectSleepStatesDataset",
    "NpzDataset",
    "M4Dataset",
]
