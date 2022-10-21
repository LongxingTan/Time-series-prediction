"""Demo of time series prediction by tfts"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from examples.utils import set_seed
import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, required=False, help="seed")
    parser.add_argument("--use_model", type=str, default="transformer", help="model for train")
    parser.add_argument("--use_data", type=str, default="airpassengers", help="dataset: sine or airpassengers")
    parser.add_argument("--train_length", type=int, default=12, help="sequence length for input")
    parser.add_argument("--predict_length", type=int, default=12, help="sequence length for output")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate for training")

    return parser.parse_args()


def run_train(args):
    train, valid = tfts.get_data(args.use_data, args.train_length, args.predict_length, test_size=0.2)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # for strong seasonality data like sine or air passengers, set up skip_connect_circle True
    custom_params = AutoConfig(args.use_model).get_config()
    custom_params.update({"skip_connect_circle": True})

    model = AutoModel(args.use_model, predict_length=args.predict_length)

    trainer = KerasTrainer(model, optimizer=optimizer, loss_fn=loss_fn)
    trainer.train(train, valid, n_epochs=args.n_epochs, early_stopping=EarlyStopping("val_loss", patience=5))

    pred = trainer.predict(valid[0])
    trainer.plot(history=valid[0], true=valid[1], pred=pred)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run_train(args)
    plt.show()
