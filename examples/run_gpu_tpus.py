# this script provides an example to use multi-gpus or tpu to train the model

import tensorflow as tf

from tfts.model import AutoModel
from tfts.trainer import KerasTrainer


def main():
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = AutoModel("seq2seq", predict_sequence_length=10).build_model([])
        trainer = KerasTrainer(model)
        trainer.train(0)
