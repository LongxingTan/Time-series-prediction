

import tensorflow as tf
import functools
from data.prepare_data import PassengerData


class DataLoader(object):
    def __init__(self,data_dir):
        self.data_dir=data_dir

    def __call__(self,batch_size,training):
        prepare_data = PassengerData(self.data_dir)

        dataset = tf.data.Dataset.from_tensor_slices(prepare_data.get_examples(self.data_dir))
        if training:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size).prefetch(5)
        return dataset

