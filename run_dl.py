import tensorflow as tf
from models.lstm import build_lstm_input_fn,build_lstm_model_fn
from models.seq2seq import build_seq_input_fn,build_seq_model_fn
from config import params
from create_model_input import Input_builder
import pandas as pd
import numpy as np
from create_features import Create_features

def run_pred():
    model_fn=build_lstm_model_fn(params)
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=180)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=params["model_dir"],config=run_config)

    if params['do_train']:
        examples = pd.read_csv(params['calendar_data'])
        x, y = Input_builder().create_RNN_input(examples.iloc[:, -1].values, train_window=params['input_seq_length'])
        train_input_fn = build_lstm_input_fn(x, y, params, is_training=True)
        estimator.train(train_input_fn)



def run_seq():
    model_fn=build_seq_model_fn(params)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=params['model_dir'])
    examples=pd.read_csv(params['calendar_data']).T
    data = examples.values[1:,:]
    features= Create_features()(examples)
    train_input_fn=build_seq_input_fn(data,features,params,is_training=True)
    estimator.train(train_input_fn)


if __name__=='__main__':
    params.update({'lstm_hidden_size':100})
    run_seq()
