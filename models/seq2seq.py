import tensorflow as tf
import logging

logging.getLogger("tensorflow").setLevel(logging.INFO)
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



def build_seq_input_fn(data,features,params,is_training=True):
    def split_and_mask():
        # split the train and eval, mask the target for eval
        train_window,predict_window=int(data.shape[1]*0.7),int(data.shape[1]*0.3)
        input_encoder=features[:,:train_window,:]
        input_decoder=features[:,predict_window:,1:]
        output_target=data[:,predict_window:]
        return input_encoder,input_decoder,output_target

    def extend():
        pass


    def input_fn():
        if is_training:
            input_encoder, input_decoder, output_target=split_and_mask()
            print(input_encoder.shape,input_decoder.shape,output_target.shape)
        else:
            input_encoder, input_decoder, output_target =extend()
        num_examples,train_window,n_features=input_encoder.shape
        num_examples, predict_window, n_predict_features = input_decoder.shape

        d=tf.data.Dataset.from_tensor_slices({
            'input_encoder':tf.constant(input_encoder,shape=[num_examples,train_window,n_features],dtype=tf.float32),
            'input_decoder':tf.constant(input_decoder,shape=[num_examples,predict_window, n_predict_features ],dtype=tf.float32),
            'output_target':tf.constant(output_target,shape=[num_examples,predict_window],dtype=tf.float32)})
        if is_training:
            d=d.repeat()
            d=d.shuffle(buffer_size=500)
        d=d.batch(batch_size=params['batch_size'])
        return d
    return input_fn


def build_seq_model_fn(params):
    def model_fn(features,labels,mode):
        encoder_inputs=features['input_encoder']
        decoder_inputs=features['input_decoder']
        decoder_targets=features['output_target']

        model=Seq2seq(params,mode)
        decoder_outputs=model(encoder_inputs,decoder_inputs)

        if mode==tf.estimator.ModeKeys.PREDICT:
            predict_sequence=0
            predictions={'predict_sequence':predict_sequence}
            return tf.estimator.EstimatorSpec(mode,predictions=predictions)
        else:
            loss=tf.losses.mean_squared_error(decoder_targets,decoder_outputs)
            if mode==tf.estimator.ModeKeys.EVAL:
                metrics={}
                return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
            else:
                train_op=tf.train.AdamOptimizer(learning_rate=params['learning_rate'])\
                    .minimize(loss,global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    return model_fn
