
from models.seq2seq import Seq2seq
import tensorflow as tf
from config import params
from prepare_data2 import Prepare_Mercedes_calendar
import logging


logging.getLogger("tensorflow").setLevel(logging.INFO)

def build_input_fn(mode,batch_size):
    mercedes = Prepare_Mercedes_calendar(failure_file='./raw_data/failures')
    data = mercedes.failures_aggby_calendar.values
    features = mercedes.features

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
        if mode=='train':
            input_encoder, input_decoder, output_target=split_and_mask()
        if mode=='predict':
            input_encoder, input_decoder, output_target =extend()
        num_examples,train_window,n_features=input_encoder.shape
        num_examples, predict_window, n_predict_features = input_decoder.shape

        d=tf.data.Dataset.from_tensor_slices({
            'input_encoder':tf.constant(input_encoder,shape=[num_examples,train_window,n_features],dtype=tf.float32),
            'input_decoder':tf.constant(input_decoder,shape=[num_examples,predict_window, n_predict_features ],dtype=tf.float32),
            'output_target':tf.constant(output_target,shape=[num_examples,predict_window],dtype=tf.float32)})
        if mode=='train':
            d=d.repeat()
            d=d.shuffle(buffer_size=500)
        d=d.batch(batch_size=batch_size)
        return d
    return input_fn


def build_model_fn(params):
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



def run_prediction():
    model_fn=build_model_fn(params)
    run_config=tf.estimator.RunConfig(save_checkpoints_secs=180)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=params['model_dir'],config=run_config)

    if params['do_train']:
        train_input_fn=build_input_fn(mode='train',batch_size=params['batch_size'])
        estimator.train(input_fn=train_input_fn,max_steps=200)


if __name__ == '__main__':
    run_prediction()