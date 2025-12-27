import argparse
import math
import random

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import tensorflow as tf

from tfts import AutoConfig, AutoModel, Pipeline, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="tfts forecasting")
    parser.add_argument("--task_id", type=str, required=True, help="which task, 1, 2, 3")
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
    forecaster = Pipeline(cfg)

    forecaster.train()


if __name__ == "__main__":
    main()
