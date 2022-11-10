"""Generate the example data script"""

import logging
import random

import numpy as np
import pandas as pd

air_passenger_url = (
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)


def get_data(name: str = "sine", train_length: int = 24, predict_length: int = 8, test_size: float = 0.1):
    assert (test_size >= 0) & (test_size <= 1), "test_size is the ratio of test dataset"
    if name == "sine":
        return get_sine(train_length, predict_length, test_size=test_size)

    elif name == "airpassengers":
        return get_air_passengers(train_length, predict_length, test_size=test_size)

    else:
        raise ValueError("unsupported data of {} yet, try 'sine', 'airpassengers'".format(name))


def get_sine(train_sequence_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.2, n_examples=100):
    x = []
    y = []
    for _ in range(n_examples):
        rand = random.random() * 2 * np.pi
        sig1 = np.sin(np.linspace(rand, 3.0 * np.pi + rand, train_sequence_length + predict_sequence_length))
        sig2 = np.cos(np.linspace(rand, 3.0 * np.pi + rand, train_sequence_length + predict_sequence_length))

        x1 = sig1[:train_sequence_length]
        y1 = sig1[train_sequence_length:]
        x2 = sig2[:train_sequence_length]
        y2 = sig2[train_sequence_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])

        x.append(x_.T)
        y.append(y_.T)

    x = np.array(x)[:, :, 0:1]
    y = np.array(y)[:, :, 0:1]
    logging.info("Load sine data", x.shape, y.shape)

    if test_size > 0:
        slice = int(n_examples * (1 - test_size))
        x_train = x[:slice]
        y_train = y[:slice]
        x_valid = x[slice:]
        y_valid = y[slice:]
        return (x_train, y_train), (x_valid, y_valid)
    return x, y


def get_air_passengers(train_sequence_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.2):
    """Air passengers data, just use divide 500 to normalize it"""
    # air_passenger_url = "../examples/data/international-airline-passengers.csv"
    df = pd.read_csv(air_passenger_url, parse_dates=None, date_parser=None, nrows=144)
    v = df.iloc[:, 1:2].values
    v = (v - np.max(v)) / (np.max(v) - np.min(v))  # MinMaxScaler

    x, y = [], []
    for seq in range(1, train_sequence_length + 1):
        x_roll = np.roll(v, seq, axis=0)
        x.append(x_roll)
    x = np.stack(x, axis=1)
    x = x[train_sequence_length:-predict_sequence_length, ::-1, :]

    for seq in range(predict_sequence_length):
        y_roll = np.roll(v, -seq)
        y.append(y_roll)
    y = np.stack(y, axis=1)
    y = y[train_sequence_length:-predict_sequence_length]
    logging.info("Load air passenger data", x.shape, y.shape)

    if test_size > 0:
        slice = int(len(x) * (1 - test_size))
        x_train = x[:slice]
        y_train = y[:slice]
        x_valid = x[slice:]
        y_valid = y[slice:]
        return (x_train, y_train), (x_valid, y_valid)
    return x, y
