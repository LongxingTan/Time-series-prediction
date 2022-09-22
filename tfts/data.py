"""
Get the example data script
"""

import os
import random
import numpy as np


# os.run(
#     "wget https://www.kaggle.com/andreazzini/international-airline-passengers/download -O data/international-airline-passengers.zip"
# )


def load(name="sine", split=0.1):
    assert (split >= 0) & (split <= 1), "split is the ratio of valid dataset"
    if name == "sine":
        return load_sine(split=split)

    return


def load_sine(split):
    n_examples = 100
    sequence_length = 24
    predict_sequence_length = 8

    x = []
    y = []
    for _ in range(n_examples):
        rand = random.random() * 2 * np.pi
        sig1 = np.sin(np.linspace(rand, 3. * np.pi + rand, sequence_length + predict_sequence_length))
        sig2 = np.cos(np.linspace(rand, 3. * np.pi + rand, sequence_length + predict_sequence_length))

        x1 = sig1[:sequence_length]
        y1 = sig1[sequence_length:]
        x2 = sig2[:sequence_length]
        y2 = sig2[sequence_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])

        x.append(x_.T)
        y.append(y_.T)

    x = np.array(x)
    y = np.array(y)[:, :, 0:1]
    print(x.shape, y.shape)

    if split > 0:
        slice = int(n_examples * (1 - split))
        x_train = x[: slice]
        y_train = y[: slice]
        x_valid = x[slice:]
        y_valid = y[slice:]
        return (x_train, y_train), (x_valid, y_valid)
    return x, y
