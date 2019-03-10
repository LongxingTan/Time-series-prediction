import tensorflow as tf

class Time_Seq2seq(object):
    def __init__(self,sess,config):
        self.sess = sess
        self.config = config
        self.weight = tf.get_variable(name='weight_out', shape=[self.config.lstm_hidden_size, self.config.output_dim],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        self.bias = tf.get_variable('bias_out', shape=[self.config.output_dim], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.))
        self.mode = None

    def build(self):
        self._build_init()
        self._build_encoder()
        self._build_decoder()
        #self.summory_op=tf.summary.merge_all()

        reshaped_outputs = [tf.matmul(i, self.weight) + self.bias for i in self.decoder_outputs]
        print([i.get_shape().as_list() for i in reshaped_outputs])

        with tf.variable_scope('Loss'):
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, self.decoder_targets):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))
            print(output_loss)
            self.loss = output_loss

        with tf.variable_scope('Optimizer'):
            #clip_gradients
            #self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                        learning_rate=self.config.learning_rate,
                                                        global_step=self.global_step,
                                                        optimizer='Adam',
                                                        clip_gradients=2.5)


    def _build_init(self):
        self.encoder_inputs=[tf.placeholder(dtype=tf.float32,shape=(None,self.config.input_dim),name='input_x_{}'.format(t))
                             for t in range(self.config.input_seq_length)]
        self.decoder_targets=[tf.placeholder(dtype=tf.float32,shape=(None,self.config.output_dim),name="input_y_{}".format(t))
                              for t in range(self.config.output_seq_length)]
        #guider training or unguided training
        self.decoder_inputs=[tf.zeros_like(self.decoder_targets[0],dtype=tf.float32,name='GO')]+self.decoder_targets[:-1]
        print(len(self.decoder_inputs),[i.get_shape().as_list() for i in self.decoder_inputs])
        self.global_step=tf.train.get_or_create_global_step()

    def _build_encoder(self):
        print(len(self.encoder_inputs),[i.get_shape().as_list() for i in self.encoder_inputs])
        with tf.variable_scope("encoder"):
            cell = self._build_encoder_cell()
            _,self.encoder_last_state = tf.contrib.rnn.static_rnn(cell, self.encoder_inputs, dtype=tf.float32)


    def _build_decoder(self):
        with tf.variable_scope("decoder"):
            cell=self._build_decoder_cell()
            state = self.encoder_last_state
            self.decoder_outputs = []
            prev = None

            for i, input in enumerate(self.decoder_inputs):
                if self.mode!='train' and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        input = self._loop_function(prev, i)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    print('input',input.get_shape().as_list())
                output, state = cell(input, state)
                self.decoder_outputs.append(output)
                if self.mode!='train':
                    prev = output
        print(len(self.decoder_outputs),[i.get_shape().as_list() for i in self.decoder_outputs])
        #print(len(self.state),[i.get_shape().as_list() for i in self.state])


    def _loop_function(self,prev, _):
        return tf.matmul(prev,self.weight) + self.bias

    def _build_encoder_cell(self):
        with tf.variable_scope('encoder_cell'):
            cells = []
            for i in range(self.config.num_stacked_layers):
                with tf.variable_scope('encoder_lstm_{}'.format(i)):
                    cells.append(tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size))
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return multi_cell

    def _build_decoder_cell(self):
        with tf.variable_scope('decoder_cell'):
            cells = []
            for i in range(self.config.num_stacked_layers):
                with tf.variable_scope('decoder_lstm_{}'.format(i)):
                    cells.append(tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size))
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return multi_cell

    def train(self,x,y):
        self.mode='train'
        self.build()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        feed_dict={self.encoder_inputs[t]:x[:,t].reshape(-1,self.config.input_dim) for t in range(self.config.input_seq_length)}
        feed_dict.update({self.decoder_targets[t]:y[:,t].reshape(-1,self.config.output_dim) for t in range(self.config.output_seq_length)})

        for i in range(self.config.n_epochs):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            print(loss)
        self.saver.save(self.sess, './result/checkpoint/seq2seq.ckpt')

    def eval(self):
        pass

    def predict(self):
        pass

    def plot(self):
        pass