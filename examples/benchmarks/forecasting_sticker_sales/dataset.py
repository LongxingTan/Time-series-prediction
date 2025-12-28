import gc
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

from joblib import Parallel, delayed
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

warnings.filterwarnings("ignore")


# https://www.kaggle.com/code/cdeotte/transformer-starter-lb-0-052
class DataReader:
    def __init__(self, use_internet=True, path="./"):
        self.use_internet = use_internet
        self.path = path
        self.means = {}
        self.stds = {}

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def add_features(self, df):
        # 1. GDP Features
        alpha3s = ["CAN", "FIN", "ITA", "KEN", "NOR", "SGP"]
        countries = np.sort(df["country"].unique())
        country_map = dict(zip(countries, alpha3s))
        df["alpha3"] = df["country"].map(country_map)
        df["year"] = df["date"].dt.year

        years = np.sort(df["year"].unique())

        if self.use_internet:
            gdp_data = {}
            for a3 in alpha3s:
                url = f"https://api.worldbank.org/v2/country/{a3}/indicator/NY.GDP.PCAP.CD?date={years[0]}:{years[-1]}&format=json"  # noqa: E501
                res = requests.get(url).json()[1]
                for entry in res:
                    gdp_data[(a3, int(entry["date"]))] = entry["value"]
            df["GDP"] = df.apply(lambda x: gdp_data.get((x["alpha3"], x["year"])), axis=1)
        else:
            gdp = pd.read_csv(f"{self.path}gdp0.csv").set_index("Unnamed: 0")
            gdp.columns = gdp.columns.astype(int)
            df["GDP"] = df.apply(lambda s: gdp.loc[s["alpha3"], s["year"]], axis=1)

        # 2. Normalization / Ratios
        df["num_sold"] /= df["GDP"]
        store_ratio = df.groupby("store")["num_sold"].transform("mean")
        df["num_sold"] /= store_ratio

        return df.drop(["alpha3", "year"], axis=1)

    def reshape_to_tensor(self, df):
        countries = list(df.country.unique())
        stores = list(df.store.unique())
        products = list(df["product"].unique())

        # Dimensions: (Products: 5, Days: 2557, Series: 18)
        # 18 series = 6 countries * 3 stores
        data = np.zeros((len(products), 2557, len(countries) * len(stores)))

        for p_idx, p_val in enumerate(products):
            for s_idx, s_val in enumerate(stores):
                for c_idx, c_val in enumerate(countries):
                    series_idx = s_idx * len(countries) + c_idx
                    f = 1.15 if c_val == "Kenya" else 1.0  # Fudge factor

                    subset = df[(df["product"] == p_val) & (df["store"] == s_val) & (df["country"] == c_val)]
                    data[p_idx, :, series_idx] = subset["num_sold"].values * f

        return data


class TrainDataset(Sequence):
    def __init__(
        self,
        data,
        product_idx=0,
        train_sequence_length=1440,
        predict_sequence_length=32,
        target_column="",
        batch_size=32,
    ):
        nans = np.isnan(data).astype("float32")
        self.data = np.stack([np.nan_to_num(data), nans], axis=-1)

        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.target_column = target_column
        self.product_idx = product_idx
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(32 * 1024 / self.batch_size))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds for the dataset")

        X = np.zeros((self.batch_size, self.train_sequence_length, 2), dtype="float32")
        y = np.zeros((self.batch_size, self.predict_sequence_length), dtype="float32")

        for i in range(self.batch_size):
            # Randomly select a series (country/store combo) and a starting time
            r = np.random.randint(0, self.data.shape[2])
            a = np.random.randint(0, self.data.shape[1] - self.train_sequence_length - self.predict_sequence_length)

            # Target is the first channel (index 0)
            target_slice = self.data[
                self.product_idx,
                a + self.train_sequence_length : a + self.train_sequence_length + self.predict_sequence_length,
                r,
                0,
            ]

            # Simple retry logic if target has NaNs (only if data is very sparse)
            X[i, :, :] = self.data[self.product_idx, a : a + self.train_sequence_length, r, :]
            y[i, :] = target_slice

        return X, y


if __name__ == "__main__":
    data_reader = DataReader()
    train_df = data_reader.load_data("/kaggle/input/playground-series-s5e1/train.csv")
    train_df = data_reader.add_features(train_df)
    data_tensor = data_reader.reshape_to_tensor(train_df)

    train_dataset = TrainDataset(
        data=data_tensor, product_idx=0, batch_size=64, train_sequence_length=1440, predict_sequence_length=32
    )
    print(train_dataset[0])
