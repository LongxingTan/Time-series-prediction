"""
This is an example of time series prediction by tfts
- multi-step prediction task
"""

from config import parse_args
from dataset import AutoData
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from utils import set_seed

from tfts import AutoConfig, AutoModel, KerasTrainer


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


def run_train():
    train_loader, valid_loader = AutoData("passenger")
    model = build_model("wavenet")
    model.fit(train_loader, valid_loader)


# if __name__ == "__main__":
#     args = parse_args()
#     set_seed(args.seed)
#     run_train(args)
