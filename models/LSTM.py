import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils import *

class Config:
    time_state=5
    hidden_size=[8]
    learning_rate=10e-3
    n_epochs=15
    batch_size=1
    n_layers=1


class LSTM():
    def __init__(self):
        self.input_x=tf.placeholder(dtype=tf.float32,shape=(None,config.time_state,1),name='input_x')
        self.input_y=tf.placeholder(dtype=tf.float32,shape=(None,1),name='input_y')
        self.dropout_keep_prob=tf.placeholder(dtype=tf.float32)
        self.hidden_size=config.hidden_size
        self.n_layers=config.n_layers
        self.time_sate=config.time_state
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=session_conf)

    def build(self):
        input = self.input_x
        for i in range(self.n_layers):
            with tf.variable_scope('rnn_%d' % i):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size[i])
                multi_layer_cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1)
                # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_out, _ = tf.nn.dynamic_rnn(multi_layer_cell, input, dtype=tf.float32)
                input = lstm_out

        #lstm_out_reshape=tf.reshape(lstm_out,[-1,self.time_sate,self.lstm_size[-1]]) #can be ignored
        #lstm_out_last=tf.gather(tf.transpose(lstm_out_reshape,[1,0,2]),self.time_sate-1)
        lstm_out_last=lstm_out[:,-1,:]

        with tf.name_scope('output'):
            w_out = tf.Variable(tf.random_uniform(shape=(self.hidden_size[-1], 1), dtype=tf.float32))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]), name='b_out')
            self.outputs = tf.nn.xw_plus_b(lstm_out_last, w_out, b_out, name='outputs')

        with tf.name_scope('loss'):
            losses = tf.reduce_sum(tf.square(self.outputs-self.input_y))
            self.loss = tf.reduce_mean(losses)


    def train(self,x_train,y_train,n_epochs=1000):
        self.build()

        with self.sess:
            #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            global_step = tf.Variable(0, name='step', trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate=10e-3).minimize(self.loss, global_step=global_step)
            self.saver=tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

            for i in range(n_epochs):
                _, step, loss = self.sess.run([train_op, global_step, self.loss], feed_dict={self.input_x: x_train,
                                                                                        self.input_y: y_train,
                                                                                        self.dropout_keep_prob: 1.0})
                print('step {}, loss {}'.format(step, loss))
            self.saver.save(self.sess, './tf_rnn.ckpt')

    def eval(self):
        pass

    def predict(self):
        pass


    def predict_point(self,x_test):
        self.load_model()
        output=self.sess.run(self.outputs,feed_dict={self.input_x:x_test,self.dropout_keep_prob:1.0})
        return output

    def predict_multi(self,x_data,predict_steps):
        '''Univarite multi-time step prediction'''
        self.load_model()
        x_data=x_data[-1,:,:]
        #x_data=tf.expand_dims(x_data,0)
        x_data=x_data[np.newaxis,:,:]

        predicted=[]
        for i in range(predict_steps):
            output=self.sess.run(self.outputs,feed_dict={self.input_x:x_data,self.dropout_keep_prob:1.0})
            predicted.append(output[0][0])
            output=tf.expand_dims(output,0)
            x_data=self.sess.run(tf.concat([x_data,output],axis=1)[:,1:,:])
        print(predicted)
        return np.array(predicted)

    def load_model(self):
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        print(" [*] Loading checkpoints...")
        self.saver.restore(self.sess, './tf_rnn.ckpt')

    def plot(self):
        pass


if __name__=='__main__':
    config = Config()
    ts = TS()

    '''
    data = ts.import_data('./Data/Repair_list.xlsx')
    # production_data = ts.import_data('Production_list.xlsx')
    calendar_data = ts.create_calendar_time(data)
    sample = calendar_data.loc[calendar_data['Fault location'] == '54027', ['Repair month', 'Value']][:-1]
    sample2 = sample.values

    x, y = ts.create_model_input(sample2, time_state=5)
    lstm = LSTM()
    lstm.train(x, y)
    lstm.predict_multi(x, predict_steps=20)

'''
    dataset = pd.read_csv('./Data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3,
                              sep=';')
    dataset = dataset.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))
    trainX, trainY = ts.create_model_input(train, time_state=5)
    testX, testY=ts.create_model_input(test,time_state=5)



    lstm = LSTM()
    lstm.train(trainX, trainY)
    trainPredict = lstm.predict_point(trainX)
    testPredict = lstm.predict_point(testX)

    trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
    testY = scaler.inverse_transform(testY)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[config.time_state:len(trainPredict) + config.time_state, :] = trainPredict
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] =np.nan
    testPredictPlot[len(trainPredict) + (config.time_state * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
