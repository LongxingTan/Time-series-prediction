# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper:
# other implementations:

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRUCell, LSTMCell, RNN

params={
    'rnn_size':64,
    'dense_size':16,
    'num_stacked_layers':1,
}


class Seq2seq(object):
    def __init__(self, custom_model_params):
        params.update(custom_model_params)
        self.encoder=Encoder(params)
        self.decoder=Decoder(params)

    def __call__(self, x, predict_seq_length,training):
        encoder_output,encoder_state=self.encoder(x)
        decoder_output = self.decoder(None,encoder_output,encoder_state, x, predict_seq_length=predict_seq_length)
        return decoder_output


class Encoder(object):
    def __init__(self,params):
        self.params=params
        cell = GRUCell(units=self.params['rnn_size'])
        self.rnn = RNN(cell,return_state=True,return_sequences=True)
        self.dense = Dense(units=self.params['dense_size'])

    def __call__(self, inputs, training=None, mask=None):
        outputs, state = self.rnn(inputs)  # outputs: batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        #encoder_hidden_state=tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        outputs = self.dense(outputs)  # => batch_size * input_seq_length * dense_size
        return outputs,state

    def initialize_hidden_state(self):
        return tf.zeros()


class Decoder(object):
    def __init__(self,params):
        self.params = params
        self.rnn_cell = GRUCell(self.params['rnn_size'])
        self.rnn = RNN(self.rnn_cell,return_state=True, return_sequences=True)
        self.dense = Dense(units=1)

    def forward(self,decoder_inputs,encoder_outputs,init_state,decoder_init_value,predict_seq_length):
        def cond_fn(time,prev_output,prev_state,decoder_output_ta):
            return time<predict_seq_length

        def body(time,prev_output,prev_state,decoder_output_ta):
            this_input=prev_output
            if decoder_inputs is not None:
                this_feature=decoder_inputs[:,time,:]
                this_input=tf.concat([this_input,this_feature],axis=1)

            this_output,this_state=self.rnn_cell(this_input,prev_state)
            project_output=self.dense(this_output)
            decoder_output_ta=decoder_output_ta.write(time,project_output)
            return time+1,project_output,this_state,decoder_output_ta

        loop_init=[tf.constant(0,dtype=tf.int32),  # steps
                   decoder_init_value,  # decoder each step
                   init_state,  # state
                   tf.TensorArray(dtype=tf.float32,size=predict_seq_length)]
        _,_,_,decoder_outputs_ta=tf.while_loop(cond_fn,body,loop_init)

        decoder_outputs=decoder_outputs_ta.stack()
        decoder_outputs=tf.transpose(decoder_outputs,[1,0,2])
        #decoder_outputs=decoder_outputs
        return decoder_outputs

    def __call__(self, decoder_inputs, encoder_outputs,encoder_state, encoder_inputs, training=None, mask=None, predict_seq_length=1):
        #context_vector=self.attention(x)
        #x=tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)
        #output,state=self.rnn(x)
        #x=self.dense(output)
        return self.forward(decoder_inputs=decoder_inputs,
                            encoder_outputs=encoder_outputs,
                            init_state=[encoder_state],  # for tf2
                            decoder_init_value=tf.expand_dims(encoder_inputs[:,-1,0],1),  # Adjust it by your own data!!!
                            predict_seq_length=predict_seq_length)


class Attention(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
