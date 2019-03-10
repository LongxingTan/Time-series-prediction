import tensorflow as tf
from prepare_data2 import Prepare_Mercedes_calendar

mercedes=Prepare_Mercedes_calendar(failure_file = './raw_data/failures')
data=mercedes.failures_aggby_calendar.values
features=mercedes.features

root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)

