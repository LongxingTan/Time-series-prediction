import gc
import glob
import os
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


def read_data():
    return


class TrainDataset(Sequence):
    def __init__(self, data, train_sequence_length, predict_sequence_length, target_column, batch_size):
        self.data = data
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.target_column = target_column
        self.batch_size = batch_size

    def __len__(self):
        return

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds for the dataset")
        return
