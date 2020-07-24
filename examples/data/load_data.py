
import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import functools
import tensorflow as tf
from data.read_data import PassengerData, SineData


class DataLoader(object):
    def __init__(self, use_dataset='passenger'):
        if use_dataset == 'passenger':
            self.data_reader = PassengerData
        elif use_dataset == 'sine':
            self.data_reader = SineData

    def __call__(self, params, data_dir, batch_size, training, sample=1):
        data_reader = self.data_reader(params)
        dataset = tf.data.Dataset.from_tensor_slices(data_reader.get_examples(data_dir,sample=sample))
        if training:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__=='__main__':
    data_loader=DataLoader('sine')
    dataset=data_loader(params={}, data_dir=None, batch_size=8, training=True)
    print(dataset.take(1))
