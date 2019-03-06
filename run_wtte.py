import tensorflow as tf
from models.WTTE import Time_WTTE
import numpy as np

class Config(object):
    lstm_hidden_size = 64
    n_epochs = 10
    n_state = None
    n_feature = None
    learning_rate=10e-4



from sklearn.preprocessing import normalize

def load_file(name):
    with open(name, 'r') as file:
        return np.loadtxt(file, delimiter=',')

np.set_printoptions(suppress=True, threshold=10000)

train = load_file('./raw_data/jet_engine_train.csv')
test_x = load_file('./raw_data/jet_engine_test_x.csv')
test_y = load_file('./raw_data/jet_engine_test_y.csv')

# Combine the X values to normalize them, then split them back out
all_x = np.concatenate((train[:, 2:26], test_x[:, 2:26]))
all_x = normalize(all_x, axis=0)

train[:, 2:26] = all_x[0:train.shape[0], :]
test_x[:, 2:26] = all_x[train.shape[0]:, :]

# Make engine numbers and days zero-indexed, for everybody's sanity
train[:, 0:2] -= 1
test_x[:, 0:2] -= 1

# Configurable observation look-back period for each engine/day
max_time = 100

def build_data(engine, time, x, max_time, is_test):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)

    for i in range(100):
        print("Engine: " + str(i))
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x = np.concatenate((this_x, xtemp))

        out_x = np.concatenate((out_x, this_x))

    return out_x, out_y


train_x, train_y = build_data(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
#print(train_x.shape)


def run_prediction():
    #session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess=tf.Session()
    config=Config()

    config.n_state=train_x.shape[1]
    config.n_feature=train_x.shape[2]
    model=Time_WTTE(session=sess,config=config)
    model.train(train_x,train_y)

run_prediction()


