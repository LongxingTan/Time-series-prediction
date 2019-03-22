import tensorflow as tf
import numpy as np


class Time_LSTM():
    def __init__(self,sess,config):
        self.sess=sess
        self.config=config

    def build(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=(None, self.config.n_states, self.config.n_features), name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        input = self.input_x
        for i in range(self.config.n_layers):
            with tf.variable_scope('rnn_%d' % i):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size[i])
                multi_layer_cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1)
                # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_out, _ = tf.nn.dynamic_rnn(multi_layer_cell, input, dtype=tf.float32)
                input = lstm_out

        #lstm_out_reshape=tf.reshape(lstm_out,[-1,self.time_sate,self.lstm_size[-1]]) #can be ignored
        #lstm_out_last=tf.gather(tf.transpose(lstm_out_reshape,[1,0,2]),self.time_sate-1)
        lstm_out_last=lstm_out[:,-1,:]

        with tf.name_scope('output'):
            w_out = tf.Variable(tf.random_uniform(shape=(self.config.hidden_size[-1], 1), dtype=tf.float32))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]), name='b_out')
            self.outputs = tf.nn.xw_plus_b(lstm_out_last, w_out, b_out, name='outputs')

        with tf.name_scope('loss'):
            losses = tf.reduce_sum(tf.square(self.outputs-self.input_y))
            self.loss = tf.reduce_mean(losses)
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,global_step=self.global_step)


    def train(self,x_train,y_train,mode=None,restore=None):
        self.build()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if mode != 'continue':
            tf.logging.info('Model building ...')
        else:
            if restore:
                tf.logging.info('Model continuing ...')

        for i in range(self.config.n_epochs):
            _, step, loss = self.sess.run([self.train_op, self.global_step, self.loss],
                                          feed_dict={self.input_x: x_train,self.input_y: y_train,self.dropout_keep_prob: 1.0})
            print('step {}, loss {}'.format(step, loss))
        self.saver.save(self.sess, './result/checkpoint/lstm.ckpt')


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
        #self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        print(" [*] Loading checkpoints...")
        self.saver.restore(self.sess, './result/checkpoint/lstm.ckpt')

    def plot(self):
        pass