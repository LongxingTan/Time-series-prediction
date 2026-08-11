"""Demo of time series prediction by tfts

Two equivalent approaches:
  1. Simple pipeline API (recommended)
     python run_prediction_simple.py --use_model dlinear
  2. Manual API (for full control, original style)
     python run_prediction_simple.py --use_model rnn --manual
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import tfts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, help="seed")
    parser.add_argument("--use_model", type=str, default="dlinear", help="model for train")
    parser.add_argument("--use_data", type=str, default="sine", help="dataset: sine or air passengers")
    parser.add_argument("--train_length", type=int, default=24, help="sequence length for train")
    parser.add_argument("--predict_sequence_length", type=int, default=12, help="sequence length for predict")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--manual", action="store_true", help="Use the manual API instead of pipeline")
    return parser.parse_args()


# ==============================================================
# Approach 1: Simple Pipeline API (recommended for most users)
# ==============================================================


def run_pipeline(args):
    """3-line forecasting with the pipeline API."""
    tfts.set_seed(args.seed)

    # Get data
    train, valid = tfts.get_data(args.use_data, args.train_length, args.predict_sequence_length, test_size=0.2)

    # Create pipeline
    pipe = tfts.pipeline(
        "forecasting",
        model=args.use_model,
        lookback=args.train_length,
        horizon=args.predict_sequence_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=5,
        seed=args.seed,
    )
    pipe.summary()

    # Train (using raw arrays — pipeline handles the rest)
    _ = pipe.trainer.train(
        train,
        valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        early_stopping_patience=5,
    )

    pred = pipe.trainer.predict(valid[0])
    pipe.trainer.plot(history=valid[0], true=valid[1], pred=pred)
    return pred


# ==============================================================
# Approach 2: Manual API (full control — original style)
# ==============================================================


def run_manual(args):
    """Step-by-step control with AutoConfig / AutoModel / Trainer."""
    tfts.set_seed(args.seed)
    train, valid = tfts.get_data(args.use_data, args.train_length, args.predict_sequence_length, test_size=0.2)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    config = tfts.AutoConfig.for_model(args.use_model)
    model = tfts.AutoModel.from_config(config, predict_sequence_length=args.predict_sequence_length)

    trainer = tfts.Trainer(model)
    trainer.train(
        train,
        valid,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        callbacks=[EarlyStopping("val_loss", patience=5)],
    )

    pred = trainer.predict(valid[0])
    trainer.plot(history=valid[0], true=valid[1], pred=pred)

    # Evaluate
    metrics = trainer.evaluate(valid)
    print(f"\nEvaluation: {metrics}")
    return pred


if __name__ == "__main__":
    args = parse_args()
    if args.manual:
        run_manual(args)
    else:
        run_pipeline(args)
    plt.show()
