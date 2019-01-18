import tensorflow as tf

class Config(object):
    pass


class seq2seq(object):
    def __init__(self):
        pass

    def build(self):
        self.input_x=tf.placeholder(dtype=tf.float32,shape=(None,self.input_dim),name='input_x')
        self.input_y=tf.placeholder(dtype=tf.float32,shape=(None,self.output_dim),name="input_y")


    def train(self,x,y):
        pass

    def eval(self):
        pass

    def predict(self):
        pass

    def plot(self):
        pass

    def _encode(self,encoder_input):
        LSTMcell_encoder=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
        LSTMcell_encoder=tf.nn.rnn_cell.DropoutWrapper(LSTMcell_encoder,output_keep_prob=self.dropout_keep_prob)
        LSTM_encoder_output,encoder_state=tf.nn.dynamic_rnn(cell=LSTMcell_encoder,inputs=encoder_input)

    def _decode(self,decoder_input):
        LSTMcell_decoder=tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
        LSTMcell_decoder=tf.nn.rnn_cell.DropoutWrapper(LSTMcell_decoder,output_keep_prob=self.dropout_keep_prob)
        LSTM_decoder_output,decoder_state=tf.nn.dynamic_rnn(cell=LSTMcell_decoder,inputs=decoder_input)