import argparse
import math
import random

from dataset import DataReader, TrainDataset
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import tensorflow as tf

from tfts import AutoConfig, AutoModel, Pipeline, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="tfts forecasting")
    parser.add_argument("--config_path", type=str, default="conf.yaml", help="Path to base config file")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode")
    parser.add_argument("--is_training", type=bool, default=True, help="Whether to train or predict")
    parser.add_argument("--model_name", type=str, default=None, help="Model name, e.g., BERT, LSTM")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)

    set_seed(cfg.seed)

    data_reader = DataReader()
    train_df = data_reader.load_data("/kaggle/input/playground-series-s5e1/train.csv")
    train_df = data_reader.add_features(train_df)
    data_tensor = data_reader.reshape_to_tensor(train_df)

    train_dataset = TrainDataset(
        data=data_tensor, product_idx=0, batch_size=64, train_sequence_length=1440, predict_sequence_length=32
    )

    forecaster = Pipeline(cfg)

    forecaster.train(train_dataset=train_dataset)


if __name__ == "__main__":
    main()
