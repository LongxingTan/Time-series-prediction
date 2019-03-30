import tensorflow as tf

class Time_WTTE(object):
    def __init__(self,params,mode):
        self.params=params
        self.mode=mode

    def build(self,inputs):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
        if self.mode=='train':
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.params['dropout_keep_prob'])

        with tf.name_scope('lstm'):
            lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=inputs, dtype=tf.float32)
            lstm_out_last = lstm_out[:, -1, :]

        with tf.name_scope('out'):
            dense_out = tf.layers.dense(lstm_out_last, units=2, name='dense')
            # activate
            dense0 = tf.exp(dense_out[:, 0], name='dense0')
            dense1 = tf.log(1 + tf.exp(dense_out[:, 1]), name='dense1')
            dense0 = tf.reshape(dense0, shape=[-1, 1])
            dense1 = tf.reshape(dense1, shape=[-1, 1])
            output = tf.concat((dense0, dense1), axis=1)
            return output

    def __call__(self,inputs,*args, **kwargs):
        outputs=self.build(inputs)
        return outputs

    def create_weibull_loss_discrete(self,y_true,y_pred):
        y_=y_true[:,0]
        u_=y_true[:,1]
        a_=y_pred[:,0]
        b_=y_pred[:,1]
        hazard0=tf.pow((y_+1e-35)/a_,b_)
        hazard1=tf.pow((y_+1)/a_,b_)
        loss=-1*tf.reduce_mean(u_*tf.log(tf.exp(hazard1-hazard0)-1.0)-hazard1)
        return loss


def build_wtte_input_fn(x, y, params, is_training=True):
    def parse_function(features, labels):
        return features, labels

    def input_fn():
        features = tf.constant(x, shape=x.shape, dtype=tf.float32)
        labels = tf.constant(y, shape=y.shape, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(parse_function)

        if is_training:
            dataset = dataset.repeat().shuffle(buffer_size=1000).batch(params['batch_size'])
        else:
            dataset = dataset.batch(params['batch_size'])
        return dataset
    return input_fn


def build_wtte_model_fn(params):
    def model_fn(features, labels, mode):
        model = Time_WTTE(params, mode)
        output = model(features)

        predictions = {"predictions": output}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = model.create_weibull_loss_discrete(y_true=labels,y_pred=output)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        else:
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = tf.metrics.root_mean_squared_error(labels=labels,
                                                                     predictions=predictions['predictions'])
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return model_fn
