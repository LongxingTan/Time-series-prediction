"""Demo of time series classification"""

import argparse
import logging

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tfts import AutoConfig, AutoModelForClassification, KerasTrainer

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, required=False, help="seed")
    parser.add_argument("--use_model", type=str, default="bert", help="model for train")
    parser.add_argument("--num_labels", type=int, default=2, help="number of unique labels")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate for training")
    return parser.parse_args()


def prepare_data():
    def readucr(filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)

    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    return x_train, y_train, x_test, y_test


def run_train(args):
    x_train, y_train, x_test, y_test = prepare_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    config = AutoConfig.for_model(args.use_model)
    model = AutoModelForClassification.from_config(config, num_labels=args.num_labels)

    opt = tf.keras.optimizers.Adam(args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    trainer = KerasTrainer(model)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    trainer.train(
        (x_train, y_train),
        valid_dataset=(x_val, y_val),
        loss_fn=loss_fn,
        optimizer=opt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        metrics=["sparse_categorical_accuracy"],
        callbacks=[early_stop_callback],
    )
    # note that tfts model summary only work during training process
    print(trainer.model.summary())

    y_pred = model(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_val, y_pred_classes)
    print(cm)
    return


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
