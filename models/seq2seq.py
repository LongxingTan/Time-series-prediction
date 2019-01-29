#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
import tensorflow as tf

class Config(object):
    pass


class seq2seq(object):
    def __init__(self):
        pass

    def build(self):
        self.input_x=[tf.placeholder(dtype=tf.float32,shape=(None,self.input_dim),name='input_x_{}'.format(t)) for t in range(input_seq_length)]
        self.input_y=[tf.placeholder(dtype=tf.float32,shape=(None,self.output_dim),name="input_y_{}".format(t)) for t in range(output_seq_length)]



    def train(self,x,y):
        pass

    def eval(self):
        pass

    def predict(self):
        pass

    def plot(self):
        pass

    def _encode(self,encoder_input):

        with tf.variable_scope('LSTMcell'):
            cells=[]
            for i in range(num_stacked_layer):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.nn.rnn_cell.LSTMCell(hidden_dim))
            cell=tf.nn.rnn_cell.MultiRNNCell(cells)
            _,enc_state=tf.contrib.rnn(cell,encoder_input,dtype=tf.float32)



        #LSTMcell_encoder=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
        #LSTMcell_encoder=tf.nn.rnn_cell.DropoutWrapper(LSTMcell_encoder,output_keep_prob=self.dropout_keep_prob)
        #LSTM_encoder_output,encoder_state=tf.nn.dynamic_rnn(cell=LSTMcell_encoder,inputs=encoder_input)

    def _decode(self,decoder_input,initial_state,cell,loop_function=None):
        with tf.variable_scope('run_decoder'):
            state=initial_state
            outputs=[]
            prev=None

            for i,inp in enumerate(decoder_input):
                if loop_function is not None and prev is not None:
                    with tf.variable_scope('loop',reuse=True):
                        inp=loop_function(prev,i)
                if i>0:
                    variable_scope.get_variable_scope().reuse_variable()
                output,state=cell(inp,state)
                outputs.append(output)

                if loop_function is not None:
                    prev=output
        return  outputs,state

        #LSTMcell_decoder=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
        #LSTMcell_decoder=tf.nn.rnn_cell.DropoutWrapper(LSTMcell_decoder,output_keep_prob=self.dropout_keep_prob)
        #LSTM_decoder_output,decoder_state=tf.nn.dynamic_rnn(cell=LSTMcell_decoder,inputs=decoder_input)