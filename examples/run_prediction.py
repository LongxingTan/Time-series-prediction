"""
This is an example of time series prediction by tfts
- multi-step prediction task
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from examples.dataset import AutoData
from examples.utils import set_seed
import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_model", type=str, default="seq2seq", help="model for train, seq2seq, wavenet, transformer"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data/international-airline-passengers.csv", help="data directory"
    )
    parser.add_argument("--model_dir", type=str, default="../weights/checkpoint", help="saved checkpoint directory")
    parser.add_argument("--saved_model_dir", type=str, default="../weights", help="saved pb directory")
    parser.add_argument("--log_dir", type=str, default="../data/logs", help="saved pb directory")
    parser.add_argument("--train_length", type=int, default=20, help="sequence length for input")
    parser.add_argument("--predict_length", type=int, default=5, help="sequence length for output")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate for training")

    args = parser.parse_args()
    return args


def build_data():
    return


def build_model(use_model):
    inputs = Input()
    config = AutoConfig(use_model)
    print(config)

    backbone = AutoModel(use_model, config)
    outputs = backbone(inputs)
    model = tf.keras.Model(inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(0.003)
    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer, loss_fn)
    return model


def run_train(args):
    train, valid = tfts.get_data("sine", args.train_length, args.predict_length, test_size=0.2)
    model = build_model("wavenet")
    model.fit(train, valid)


# if __name__ == "__main__":
#     args = parse_args()
#     set_seed(args.seed)
#     run_train(args)
