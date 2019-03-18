import tensorflow as tf


class Seq2seq(object):
    def __init__(self,params,mode):
        self.params=params
        self.mode=mode

    def build(self,encoder_inputs,decoder_inputs):
        encoder_outputs, encoder_final_state = self.build_encoder(encoder_inputs=encoder_inputs)

        enc_stab_loss=self.create_stab_loss(encoder_outputs,beta=0.0)
        sequence_length=self.create_seq_length(decoder_inputs)
        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, sequence_length)

        decoder_outputs = self.build_decoder(train_helper,encoder_final_state,'decode')
        decoder_outputs=decoder_outputs.rnn_output[:,:,-1] #Todo clarify it
        #print(decoder_outputs.get_shape().as_list())
        dec_stab_loss=self.create_stab_loss(decoder_outputs,beta=0.0)
        return decoder_outputs

    def build_encoder(self,encoder_inputs):
        # => batch_size * seg_length * lstm_hidden_size
        cell = tf.contrib.rnn.GRUCell(num_units=self.params['lstm_hidden_size'])
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32)
        return encoder_outputs, encoder_final_state

    def build_decoder(self,helper,encoder_final_state, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.contrib.rnn.GRUCell(num_units=self.params['lstm_hidden_size'])
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper,initial_state=encoder_final_state)
            outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
        return outputs[0]


    def create_stab_loss(self,rnn_output, beta):
        if beta == 0.0:
            return 0.0
        l2 = tf.sqrt(tf.reduce_sum(tf.square(rnn_output), axis=-1))  # [time, batch, features] -> [time, batch]
        return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))  #  [time, batch] -> []

    def create_seq_length(self,inputs):
        used=tf.sign(tf.reduce_max(tf.abs(inputs),2))
        length=tf.reduce_sum(used,1)
        return tf.cast(length,tf.int32)

    def __call__(self,encoder_inputs,decoder_inputs):
        decoder_outputs= self.build(encoder_inputs,decoder_inputs)
        return decoder_outputs