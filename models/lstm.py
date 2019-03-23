import tensorflow as tf
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Time_LSTM(object):
    def __init__(self,params,mode):
        self.params=params
        self.mode=mode

    def build(self,inputs):
        # batch_size * sequence_length * n_features
        with tf.variable_scope('rnn_%d' % 0):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'], state_is_tuple=True)
            #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.params['num_stacked_layers'], state_is_tuple=True)
            lstm_out, final_state = tf.nn.dynamic_rnn(cells, inputs, sequence_length=None,dtype=tf.float32)

        # lstm_out_reshape=tf.reshape(lstm_out,[-1,self.time_sate,self.lstm_size[-1]]) #can be ignored
        # lstm_out_last=tf.gather(tf.transpose(lstm_out_reshape,[1,0,2]),self.time_sate-1)
        lstm_out_last = lstm_out[:, -1, :]

        with tf.name_scope('output'):
            outputs=tf.layers.dense(lstm_out_last,units=1)
        return outputs

    def __call__(self,inputs,*args, **kwargs):
        outputs=self.build(inputs)
        return outputs



def build_lstm_input_fn(x,y,params,is_training=True):
    def parse_function(features,labels):
        return features,labels

    def input_fn():
        features = tf.constant(x,shape=x.shape,dtype=tf.float32)
        labels = tf.constant(y,shape=y.shape,dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(parse_function)

        if is_training:
            dataset = dataset.repeat().shuffle(buffer_size=1000).batch(params['batch_size'])
        else:
            dataset = dataset.batch(params['batch_size'])
        return dataset
    return input_fn


def build_lstm_model_fn(params):
    def model_fn(features,labels,mode):

        model=Time_LSTM(params,mode)
        output=model(features)

        predictions={"predictions":output}
        if mode==tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

        loss=tf.losses.mean_squared_error(labels=labels,predictions=output)
        if mode==tf.estimator.ModeKeys.TRAIN:
            optimizer=tf.train.AdamOptimizer()
            train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

        else:
            if mode==tf.estimator.ModeKeys.EVAL:
                eval_metric_ops=tf.metrics.root_mean_squared_error(labels=labels,predictions=predictions['predictions'])
                return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
    return model_fn
