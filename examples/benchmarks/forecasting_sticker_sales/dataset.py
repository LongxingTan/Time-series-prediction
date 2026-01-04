import warnings

from joblib import Parallel, delayed
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence

warnings.filterwarnings("ignore")


# https://www.kaggle.com/code/cdeotte/transformer-starter-lb-0-052
class TimeSeriesProcessor:
    def __init__(self, use_internet=True, path="./"):
        self.use_internet = use_internet
        self.path = path
        self.scales = {}
        self.gdp_data = None

    def fetch_gdp(self, df):
        """Unified GDP fetching logic."""
        alpha3_map = {
            "Canada": "CAN",
            "Finland": "FIN",
            "Italy": "ITA",
            "Kenya": "KEN",
            "Norway": "NOR",
            "Singapore": "SGP",
        }
        df["alpha3"] = df["country"].map(alpha3_map)
        df["year"] = df["date"].dt.year
        years = df["year"].unique()

        if self.use_internet:
            gdp_dict = {}
            for country, a3 in alpha3_map.items():
                try:
                    url = f"https://api.worldbank.org/v2/country/{a3}/indicator/NY.GDP.PCAP.CD?date={min(years)}:{max(years)}&format=json"  # noqa: E501,E231
                    res = requests.get(url).json()[1]
                    for entry in res:
                        gdp_dict[(a3, int(entry["date"]))] = entry["value"]
                except Exception as e:
                    print(f"Error fetching GDP for {a3}: {e}")
            self.gdp_data = gdp_dict
        else:
            # Assume local file exists
            gdp_df = pd.read_csv(f"{self.path}gdp.csv").set_index("alpha3")
            self.gdp_data = gdp_df.to_dict()

        return df

    def process_features(self, df, is_train=True):
        """Calculates GDP ratios and store-based normalization."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if self.gdp_data is None:
            df = self.fetch_gdp(df)

        df["GDP"] = df.apply(lambda x: self.gdp_data.get((x["alpha3"], x["year"]), 1.0), axis=1)

        # 1. GDP Normalization
        df["scaled_target"] = df["num_sold"] / df["GDP"]

        # 2. Store Ratio (calculate during train, apply during test)
        if is_train:
            self.store_ratios = df.groupby("store")["scaled_target"].mean().to_dict()

        df["scaled_target"] /= df["store"].map(self.store_ratios)

        # 3. Kenya Fudge Factor
        df.loc[df["country"] == "Kenya", "scaled_target"] *= 1.15

        return df

    def dataframe_to_tensor(self, df):
        """
        Pivots the dataframe into a 3D tensor: (Products, Time, Series)
        Series = Country + Store combinations.
        """
        # Create a unique key for each Country/Store combination
        df["series_key"] = df["country"] + "_" + df["store"]

        products = sorted(df["product"].unique())
        series_keys = sorted(df["series_key"].unique())

        tensor_list = []
        for prod in products:
            # Efficient pivoting instead of nested loops
            subset = df[df["product"] == prod].pivot(index="date", columns="series_key", values="scaled_target")

            # Save scaling params per product
            if prod not in self.scales:
                self.scales[prod] = {"mean": subset.values.mean(), "std": subset.values.std()}

            # Standard Scale
            scaled_val = (subset.values - self.scales[prod]["mean"]) / self.scales[prod]["std"]
            tensor_list.append(scaled_val)

        return np.stack(tensor_list), products, series_keys

    def inverse_transform(self, pred, product_name, country, store, date):
        """Reverses all transformations to get the original num_sold scale."""
        # 1. Reverse Standard Scale
        val = (pred * self.scales[product_name]["std"]) + self.scales[product_name]["mean"]

        # 2. Reverse Kenya Factor
        if country == "Kenya":
            val /= 1.15

        # 3. Reverse Store Ratio
        val *= self.store_ratios[store]

        # 4. Reverse GDP
        year = pd.to_datetime(date).year
        # Note: You'd need a helper to get alpha3 from country
        alpha3 = {
            "Canada": "CAN",
            "Finland": "FIN",
            "Italy": "ITA",
            "Kenya": "KEN",
            "Norway": "NOR",
            "Singapore": "SGP",
        }[country]
        val *= self.gdp_data.get((alpha3, year), 1.0)

        return val


class TimeSeriesDataset(Sequence):
    def __init__(
        self,
        data,
        mode="train",  # "train" or "test"
        product_idx=0,
        train_sequence_length=1440,
        predict_sequence_length=32,
        batch_size=32,
    ):
        self.data = data[product_idx]  # Shape: (Time, Series)
        self.mode = mode
        self.product_idx = product_idx
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.batch_size = batch_size

        nans = np.isnan(self.data).astype("float32")
        self.combined_data = np.stack([np.nan_to_num(self.data), nans], axis=-1)

    def __len__(self):
        return int(np.ceil(self.data.shape[1] / self.batch_size))

    def __getitem__(self, idx):
        if self.mode == "train":
            return self._get_train_batch()
        else:
            return self._get_test_batch(idx)

    def _get_train_batch(self):
        X = np.zeros((self.batch_size, self.train_sequence_length, 2), dtype="float32")
        y = np.zeros((self.batch_size, self.predict_sequence_length), dtype="float32")

        for i in range(self.batch_size):
            series_idx = np.random.randint(0, self.data.shape[1])
            start = np.random.randint(0, self.data.shape[0] - self.train_sequence_length - self.predict_sequence_length)

            X[i] = self.combined_data[start : start + self.train_sequence_length, series_idx, :]
            y[i] = self.combined_data[
                start + self.train_sequence_length : start + self.train_sequence_length + self.predict_sequence_length,
                series_idx,
                0,
            ]
        return X, y

    def _get_test_batch(self, idx):
        """Returns the LAST train_len for each category for prediction."""
        start_series = idx * self.batch_size
        end_series = min((idx + 1) * self.batch_size, self.data.shape[1])
        actual_bs = end_series - start_series

        X = np.zeros((actual_bs, self.train_sequence_length, 2), dtype="float32")

        for i, s_idx in enumerate(range(start_series, end_series)):
            # Always take the very tail of the data
            X[i] = self.combined_data[-self.train_sequence_length :, s_idx, :]

        return X


if __name__ == "__main__":
    # 1. Process Data
    processor = TimeSeriesProcessor(use_internet=True)
    df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
    df_processed = processor.process_features(df_train, is_train=True)
    tensor, product_names, series_names = processor.dataframe_to_tensor(df_processed)

    # 2. Create Train Dataset for Product 0
    train_gen = TimeSeriesDataset(tensor, mode="train", product_idx=0)

    # 3. Create Test Dataset (the last window for all series in Product 0)
    test_gen = TimeSeriesDataset(tensor, mode="test", product_idx=0)

    # # 4. Predict
    # predictions = model.predict(test_gen) # (Total Series, pred_len)

    # # 5. Reverse Scaling for a specific prediction
    # raw_pred = processor.inverse_transform(
    #     pred=predictions[0, 0],
    #     product_name=product_names[0],
    #     country="Canada",
    #     store="KaggleMart",
    #     date="2026-01-01"
    # )
