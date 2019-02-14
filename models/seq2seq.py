
import tensorflow as tf


class Config(object):
    pass


class Time_Seq2seq(object):
    def __init__(self,num_stacked_layers,hidden_dim,lambda_l2_reg):
        self.num_stacked_layers=num_stacked_layers
        self.hidden_dim=hidden_dim
        self.lambda_l2_reg=0.0
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.weight = tf.get_variable('weight_out', shape=[self.hidden_dim, self.output_dim], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer())
        self.bias = tf.get_variable('bias_out', shape=[self.output_dim], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.))
        self.mode=None

    def build(self):
        self._build_init()
        self._build_encoder()
        self._build_decoder()
        self.summory_op=tf.summary.merge_all()


    def _build_init(self):
        self.encoder_inputs=[tf.placeholder(dtype=tf.float32,shape=(None,self.input_dim),name='input_x_{}'.format(t))
                             for t in range(self.input_seq_length)]
        self.decoder_outputs_true=[tf.placeholder(dtype=tf.float32,shape=(None,self.output_dim),name="input_y_{}".format(t))
                              for t in range(self.output_seq_length)]
        #guider training or unguided training
        self.decoder_inputs=[tf.zeros_like(self.decoder_outputs[0],dtype=tf.float32,name='GO')]+self.decoder_outputs[:-1]
        self.global_step=tf.train.get_or_create_global_step()

    def _build_encoder(self):
       with tf.variable_scope("encoder"):
            cell = self._build_encoder_cell()
            self.encoder_outputs,self.encoder_last_state = tf.nn.dynamic_rnn(cell, self.encoder_inputs, dtype=tf.float32)


    def _build_decoder(self):
        with tf.variable_scope("decoder"):
            cell=self._build_decoder_cell()
            self.state = self.encoder_last_state
            self.decoder_outputs = []
            prev = None

            for i, input in enumerate(self.decoder_inputs):
                if self.mode!='train' and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        input = self._loop_function(prev, i)
                if i > 0:
                    tf.variable_scope.get_variable_scope().reuse_variables()
                output, state = cell(input, self.state)
                self.decoder_outputs.append(output)
                if self.mode!='train':
                    prev = output
        return self.decoder_outputs, self.state


    def _loop_function(self,prev, _):
        return tf.matmul(prev,self.weight) + self.bias

    def _build_encoder_cell(self):
        with tf.variable_scope('encoder_Cell'):
            cells = []
            for i in range(self.num_stacked_layers):
                with tf.variable_scope('encoder_lstm_{}'.format(i)):
                    cells.append(tf.nn.rnn_cell.LSTMCell(self.num_stacked_layers))
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell

    def _build_decoder_cell(self):
        with tf.variable_scope('decoder_Cell'):
            cells = []
            for i in range(self.num_stacked_layers):
                with tf.variable_scope('decoder_lstm_{}'.format(i)):
                    cells.append(tf.nn.rnn_cell.LSTMCell(self.num_stacked_layers))
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell


    def train(self,x,y,n_epochs):
        self.input_dim, self.input_seq_length = x.shape[2],x.shape[1]
        self.output_dim, self.output_seq_length = y.shape[2],y.shape[1]

        self.build()
        reshaped_outputs = [tf.matmul(i, self.weight)+self.bias for i in self.decoder_outputs]

        with tf.variable_scope('Loss'):
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, self.decoder_outputs_true):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + self.lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=10e-3,
                global_step=self.global_step,
                optimizer='Adam',
                clip_gradients=2.5)

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        feed_dict={self.encoder_inputs[t]:x[:,t].reshape(-1,input_dim) for t in range(input_seq_length)}
        feed_dict.update({self.decoder_outputs[t]:y[:,t].reshape(-1,output_dim) for i in range(output_seq_length)})

        for i in range(n_epochs):
            _, loss = self.sess.run([optimizer, loss], feed_dict=feed_dict)
            print('step {}, loss {}'.format(self.global_step, loss))
        saver.save(self.sess, './tf_rnn.ckpt')

    def eval(self):
        pass

    def predict(self):
        pass

    def plot(self):
        pass