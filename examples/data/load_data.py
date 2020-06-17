

import tensorflow as tf
import functools
from data.read_data import PassengerData


class DataLoader(object):
    def __init__(self):
        pass

    def __call__(self,params,data_dir,batch_size,training,sample=1):
        prepare_data = PassengerData(params)

        dataset = tf.data.Dataset.from_tensor_slices(prepare_data.get_examples(data_dir,sample=sample))
        if training:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class DataParser(object):
    def __init__(self):
        pass

    def encode(self):
        pass


if __name__=='__main__':
    data_loader=DataLoader()
    dataset=data_loader()
    print(dataset.take(1))
