import tensorflow as tf

class Time_WTTE(object):
    def __init__(self,session,config):
        self.config=config
        self.sess = session

    def build(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=(None, self.config.n_state,self.config.n_feature), name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='input_y')
        self.dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='dropout')
        self.global_step=tf.Variable(0,name='global_step',trainable=False)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout_keep_prob)

        with tf.name_scope('lstm'):
            lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=self.input_x, dtype=tf.float32)
            lstm_out_last = lstm_out[:, -1, :]

        with tf.name_scope('out'):
            dense_out = tf.layers.dense(lstm_out_last, units=2, name='dense')
            # activate
            dense0 = tf.exp(dense_out[:, 0], name='dense0')
            dense1 = tf.log(1 + tf.exp(dense_out[:, 1]), name='dense1')
            dense0 = tf.reshape(dense0, shape=[-1, 1])
            dense1 = tf.reshape(dense1, shape=[-1, 1])
            self.output = tf.concat((dense0, dense1), axis=1)

        self.loss = self._weibull_loss_discrete(self.input_y, self.output)
        self.train_op=tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss,
                                                                                               global_step=self.global_step)

    def train(self,x,y,mode=None,restore=None):
        self.build()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if mode!='continue':
            tf.logging.info('Model building ...')
        else:
            if restore:
                tf.logging.info('Model continuing ...')

        for epoch in range(self.config.n_epochs):
            loss_epoch=[]
            mse_epoch=[]
            #batches=create_batch(list(zip(x,y)),batch_size=self.config.batch_size,n_epochs=5)

            feed_dict = {self.input_x: x,
                         self.input_y: y,
                         self.dropout_keep_prob: 1.0}
            _, loss= self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            loss_epoch.append(loss)
            #mse_epoch.append(mse)
            tf.logging.info('Epoch {},,Loss {}'.format(epoch, loss))

            self.saver.save(self.sess, './result/checkpoint/wtte.ckpt')


    def eval(self):
        pass

    def predict(self):
        pass


    def plot(self):
        pass

    def _weibull_loss_discrete(self,y_true,y_pred):
        y_=y_true[:,0]
        u_=y_true[:,1]
        a_=y_pred[:,0]
        b_=y_pred[:,1]

        hazard0=tf.pow((y_+1e-35)/a_,b_)
        hazard1=tf.pow((y_+1)/a_,b_)
        loss=-1*tf.reduce_mean(u_*tf.log(tf.exp(hazard1-hazard0)-1.0)-hazard1)
        return loss