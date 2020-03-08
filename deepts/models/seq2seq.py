
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense

params={
    'rnn_size':32,
    'dense_size':8,
    'num_stacked_layers':1,
    'predict_window_sizes':5,
}


class Seq2seq(object):
    def __init__(self):
        self.encoder=Encoder()
        self.decoder=Decoder()

    def __call__(self, inputs_shape,training):
        x=Input(inputs_shape)
        encoder_output,encoder_state=self.encoder(x)
        decoder_output = self.decoder(None,encoder_state,x)
        print('decoder_output',decoder_output)
        return tf.keras.Model(x,decoder_output,name='seq2seq')


class Encoder(object):
    def __init__(self):
        super(Encoder,self).__init__()
        self.params=params
        cell=tf.keras.layers.GRUCell(units=params['rnn_size'])
        self.rnn=tf.keras.layers.RNN(cell,return_state=True,return_sequences=True)
        self.dense=tf.keras.layers.Dense(units=params['dense_size'])

    def __call__(self, inputs, training=None, mask=None):
        inputs,hidden_state=self.rnn(inputs)
        #encoder_hidden_state=tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        return inputs,hidden_state

    def initialize_hidden_state(self):
        return tf.zeros()


class Decoder(object):
    def __init__(self):
        super(Decoder,self).__init__()
        self.rnn_cell=tf.keras.layers.GRUCell(params['rnn_size'])
        self.rnn=tf.keras.layers.RNN(self.rnn_cell,return_state=True,return_sequences=True)
        self.dense=tf.keras.layers.Dense(units=1)

    def forward(self,decoder_inputs,init_state,decoder_init_value):
        def cond_fn(time,prev_output,prev_state,decoder_output_ta):
            return time<params['predict_window_sizes']

        def body(time,prev_output,prev_state,decoder_output_ta):
            this_input=prev_output
            if decoder_inputs is not None:
                this_feature=decoder_inputs[:,time,:]
                this_input=tf.concat([this_input,this_feature],axis=1)

            this_output,this_state=self.rnn_cell(this_input,prev_state)
            project_output=Dense(1)(this_output)
            decoder_output_ta=decoder_output_ta.write(time,project_output)
            return time+1,project_output,this_state,decoder_output_ta

        loop_init=[tf.constant(0,dtype=tf.int32),
                   decoder_init_value,
                   init_state,
                   tf.TensorArray(dtype=tf.float32,size=params['predict_window_sizes'])]
        _,_,_,decoder_outputs_ta=tf.while_loop(cond_fn,body,loop_init)

        decoder_outputs=decoder_outputs_ta.stack()
        decoder_outputs=tf.transpose(decoder_outputs,[1,0,2])
        decoder_outputs=decoder_outputs
        return decoder_outputs

    def __call__(self, decoder_inputs,encoder_state,encoder_inputs, training=None, mask=None):
        #context_vector=self.attention(x)
        #x=tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)
        #output,state=self.rnn(x)
        #x=self.dense(output)
        return self.forward(decoder_inputs=decoder_inputs,
                            init_state=[encoder_state],  # for tf2
                            decoder_init_value=tf.expand_dims(encoder_inputs[:,-1,0],1),
                            )

