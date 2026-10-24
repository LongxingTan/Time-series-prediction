"""Load traffic observations."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ._download import TS_DATASETS_URL, download_and_extract


def load_traffic() -> pd.DataFrame:
    data_dir = download_and_extract("traffic")
    logger.info("Reading PEMS metadata files...")

    def _process_pems_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            l = [variable_type(i) for i in s.replace("[", "").replace("]", "").split()]
        else:
            l = [variable_type(i) for i in s.replace("[", "").replace("]", "").split(delimiter)]
        return l

    def _read_pems_matrix(data_folder, filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r") as dat:
            lines = dat.readlines()
            for i, line in enumerate(lines):
                # array is a list of lists (stations x time_observations)
                array = [
                    _process_pems_list(row_split, variable_type=float, delimiter=None)
                    for row_split in _process_pems_list(line, variable_type=str, delimiter=";")
                ]
                array_list.append(array)
        return array_list

    def read_single_list(fname):
        with open(os.path.join(data_dir, fname), "r") as f:
            return _process_pems_list(f.readlines()[0])

    shuffle_order = np.array(read_single_list("randperm")) - 1
    train_dayofweek = read_single_list("PEMS_trainlabels")
    test_dayofweek = read_single_list("PEMS_testlabels")
    stations_list = read_single_list("stations_list")

    logger.info("Reading and parsing train/test matrices (this may take a moment)...")
    train_tensor = _read_pems_matrix(data_dir, "PEMS_train")
    test_tensor = _read_pems_matrix(data_dir, "PEMS_test")

    # Inverse permutate shuffle order to restore temporal consistency
    inverse_mapping = {new_loc: prev_loc for prev_loc, new_loc in enumerate(shuffle_order)}
    reverse_shuffle_order = np.array([inverse_mapping[i] for i in range(len(shuffle_order))])

    # Combine and Reorder
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    logger.info("Aggregating to hourly data and formatting...")
    labels = [f"traj_{i}" for i in stations_list]
    hourly_list = []

    for day, day_matrix in enumerate(combined_tensor):
        # day_matrix.T -> index: time (144 intervals), columns: stations
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        # Sampled at 10 min intervals: 6 intervals = 1 hour
        hourly["hour_on_day"] = [int(i / 6) for i in hourly.index]

        # Mean occupancy per hour
        hourly = hourly.groupby("hour_on_day").mean()
        hourly["sensor_day"] = day
        hourly["time_on_day"] = hourly.index
        hourly["day_of_week"] = day_of_week[day]
        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True)

    # Flatten the dataframe: Each row is (sensor_id, time, occupancy)
    store_columns = [c for c in hourly_frame.columns if "traj" in c]
    other_columns = ["sensor_day", "time_on_day", "day_of_week"]

    flat_list = []
    for store in store_columns:
        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ["occupancy"] + other_columns
        sliced["station_id"] = int(store.replace("traj_", ""))

        # Calculate hours from start for a continuous time axis
        sliced["hours_from_start"] = sliced["time_on_day"] + sliced["sensor_day"] * 24.0
        flat_list.append(sliced)

    df = pd.concat(flat_list, axis=0, ignore_index=True)

    # Filter to match range used by academic papers (first 173 days)
    df = df[df["sensor_day"] < 173].copy()

    # Sorting for time-series consistency
    df = df.sort_values(["station_id", "hours_from_start"])

    return df
