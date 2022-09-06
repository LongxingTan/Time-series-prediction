
import sys
import os
filePath = os.path.abspath(os.path.dirname(''))
sys.path.append(os.path.split(filePath)[0])

import os
import unittest
import tensorflow as tf
from tfts import BertConfig


if __name__ == '__main__':
    import numpy as np
    fake_data = np.random.rand(16, 160, 35)
    rnn = RNN(custom_model_params={})
    y = rnn(tf.convert_to_tensor(fake_data, tf.float32))
    print(y.shape)
