# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# This script is a general example to load data into the model, next step is run_train.py

import functools

import tensorflow as tf

from .read_sine import PassengerData, SineData
from .read_web_traffic import WebDataReader


class DataLoader(object):
    def __init__(self, use_dataset="passenger"):
        self.use_dataset = use_dataset
        if use_dataset == "passenger":
            self.data_reader = PassengerData
        elif use_dataset == "sine":
            self.data_reader = SineData
        elif use_dataset == "web_traffic":
            self.data_reader = WebDataReader

    def __call__(self, params, data_dir, batch_size, training, sample=1):
        data_reader = self.data_reader(params)
        dataset = tf.data.Dataset.from_tensor_slices(data_reader.get_examples(data_dir, sample=sample))

        if training:
            dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class WebDataLoader(object):
    def __init__(self):
        pass

    def __call__(self, data_dir, mode, batch_size):
        data_reader = WebDataReader(data_dir, mode)
        dataset = tf.data.Dataset.from_generator(
            data_reader.iter, output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )
        if mode != "test":
            dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    data_loader = DataLoader("sine")
    dataset = data_loader(params={}, data_dir=None, batch_size=8, training=True)
    print(dataset.take(1))
