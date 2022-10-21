# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""
Data example of Kaggle WTF data
https://www.kaggle.com/c/web-traffic-time-series-forecasting
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf


def log_transform(x, sequence_mean):
    return np.log1p(x) - sequence_mean


def sequence_mean(x, effective_length):
    return np.sum(x) / effective_length


class WebDataReader(object):
    def __init__(self, data_dir, mode, train_test_ratio=0.9):
        data_cols = [
            "data",  # n_example * n_days
            "is_nan",
            "page_id",
            "project",
            "access",
            "agent",
            "test_data",
            "test_is_nan",
        ]
        self.data = [np.load(os.path.join(data_dir, "{}.npy".format(i))) for i in data_cols]
        self.n_examples = self.data[0].shape[0]
        self.mode = mode

        if mode == "test":
            self.idx = range(self.n_examples)
        elif mode == "train":
            train_idx = np.random.choice(
                range(self.n_examples), int(train_test_ratio * self.n_examples), replace=False
            )  # set p if not equal weighted sample
            self.idx = train_idx
        elif mode == "val":
            train_idx = np.random.choice(
                range(self.n_examples), int(train_test_ratio * self.n_examples), replace=False
            )  # must set fixed seed Todo: still need to check if leaks happened
            val_idx = np.setdiff1d(range(self.n_examples), train_idx)
            self.idx = val_idx
        else:
            raise ValueError("only train,test or val is valid mode")

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        x = [dat[idx] for dat in self.data]
        return self.preprocess(x)

    def iter(self):
        for i in self.idx:
            yield self[i]

    def preprocess(self, x):
        # process the saved numpy to features
        # otherwise, you can also write it in Tensorflow graph mode while tf.data.Dataset.map
        """
        output: encoder_feature: [sequence_length, n_feature]
        decoder_feature: [predict_sequence_length, decoder_n_feature]
        """

        data, nan_data, project, access, agent = x[0], x[1], x[3], x[4], x[5]
        max_encode_length = 530
        num_decode_steps = 64

        # encode feature
        x_encode = np.zeros(max_encode_length)  # x_encode: sequence_length
        is_nan_encode = np.zeros(max_encode_length)

        rand_len = np.random.randint(max_encode_length - 365 + 1, max_encode_length + 1)
        x_encode_len = max_encode_length if self.mode == "test" else rand_len
        x_encode[:x_encode_len] = data[:x_encode_len]

        log_x_encode_mean = sequence_mean(x_encode, x_encode_len)
        log_x_encode = log_transform(x_encode, log_x_encode_mean)

        is_nan_encode[:x_encode_len] = nan_data[:x_encode_len]

        project_onehot = np.zeros(9)
        np.put(project_onehot, project, 1)

        access_onehot = np.zeros(3)
        np.put(access_onehot, access, 1)

        agent_onehot = np.zeros(2)
        np.put(agent_onehot, agent, 1)

        encoder_feature = np.concatenate(
            [  # each item shape: [encode_steps, n_sub_feature]
                np.expand_dims(is_nan_encode, 1),
                np.expand_dims(np.equal(x_encode, 0.0).astype(float), 1),
                np.tile(np.expand_dims(log_x_encode, 0), [max_encode_length, 1]),
                np.tile(np.expand_dims(project_onehot, 0), [max_encode_length, 1]),
                np.tile(np.expand_dims(access_onehot, 0), [max_encode_length, 1]),
                np.tile(np.expand_dims(agent_onehot, 0), [max_encode_length, 1]),
            ],
            axis=1,
        )

        # decode feature
        decoder_feature = np.concatenate(
            [  # each item shape: [decode_steps, n_sub_feature]
                np.eye(num_decode_steps),
                np.tile(np.expand_dims(log_x_encode_mean, 0), [num_decode_steps, 1]),
                np.tile(np.expand_dims(project_onehot, 0), [num_decode_steps, 1]),
                np.tile(np.expand_dims(access_onehot, 0), [num_decode_steps, 1]),
                np.tile(np.expand_dims(agent_onehot, 0), [num_decode_steps, 1]),
            ],
            axis=1,
        )

        # decoder target
        decoder_target = np.zeros(num_decode_steps)
        is_nan_decoder_target = np.zeros(num_decode_steps)

        if not self.mode == "test":
            decoder_target = data[x_encode_len : x_encode_len + num_decode_steps]
            is_nan_decoder_target = nan_data[x_encode_len : x_encode_len + num_decode_steps]

        output = encoder_feature, decoder_feature, decoder_target, is_nan_decoder_target
        return output  # encoder_feature, decoder_feature, decoder_targets


class DataLoader(object):
    def __init__(
        self,
    ):
        pass

    def __call__(self, data_dir, mode, batch_size):
        data_reader = WebDataReader(data_dir, mode)
        dataset = tf.data.Dataset.from_generator(
            data_reader.iter, output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    # train_data_reader=DataReader(data_dir='../data/processed',mode='train')
    # train_data_reader[0]
    # val_data_reader=DataReader(data_dir='../data/processed',mode='val')
    # print(len(val_data_reader.idx))
    # test_data_reader=DataReader(data_dir='../data/processed',mode='test')
    # print(len(test_data_reader.idx))

    data_loader = DataLoader()(data_dir="../data/processed", mode="train", batch_size=2)

    for encoder_feature, decoder_feature, decoder_target, is_nan_decoder_target in data_loader.take(1):
        print(encoder_feature.shape)
        print(decoder_feature.shape)
        print(decoder_target.shape)
        print(is_nan_decoder_target.shape)
