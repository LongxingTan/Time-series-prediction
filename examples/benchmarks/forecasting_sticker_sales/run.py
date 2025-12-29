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


# def run_inference(product_idx):
#     """Runs the recursive prediction for a specific product."""
#     # Ensure history has the 2nd channel (NaN indicator)
#     # history_tensor shape: (5, 2557, 18) -> Needs expansion to (1, LEN, 18, 2)
#     data = np.expand_dims(self.history_tensor, axis=-1)
#     nans = np.isnan(data).astype('float32')
#     data = np.concatenate([data, nans], axis=-1)

#     product_preds = np.zeros((18, self.PRED_LEN * self.STEPS))
#     bad_rows = []

#     for jj in range(18):
#         # Get last window of training data for this series
#         # Shape: (1, LEN, 2)
#         current_window = data[product_idx:product_idx+1, -self.LEN:, jj, :].copy()

#         if np.isnan(current_window[:, :, 0]).sum() == self.LEN:
#             bad_rows.append(jj)
#             continue

#         series_predictions = []

#         for step in range(self.STEPS):
#             # Predict next 32 days
#             # Input shape: (1, LEN, 2)
#             p2 = self.model(np.nan_to_num(current_window))
#             p2 = p2.numpy().reshape((1, self.PRED_LEN, 1))

#             # Add dummy NaN indicator (0.0) to predictions for the next step
#             p2_with_nan = np.concatenate([p2, np.zeros_like(p2)], axis=-1)
#             series_predictions.append(p2_with_nan)

#             # Update window: Slide window forward
#             # Remove oldest 32, append newest 32
#             current_window = np.concatenate([current_window[:, self.PRED_LEN:, :], p2_with_nan], axis=1)

#         # Combine all steps and remove the NaN indicator channel
#         product_preds[jj, :] = np.concatenate([z[:, :, 0] for z in series_predictions], axis=1).flatten()

#     # Handle bad rows (series with no training data)
#     if bad_rows:
#         fill_val = np.nanmean(product_preds, axis=0)
#         for r in bad_rows:
#             product_preds[r, :] = fill_val

#     return product_preds


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
