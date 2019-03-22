from models.wtte import Time_WTTE
from models.seq2seq import Seq2seq
from models.gan import Time_GAN


def run_prediction():
    model_fn=build_model_fn(params)
    run_config=tf.estimator.RunConfig(save_checkpoints_secs=180)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=params['model_dir'],config=run_config)

    if params['do_train']:
        train_input_fn=build_input_fn(mode='train',batch_size=params['batch_size'])
        estimator.train(input_fn=train_input_fn,max_steps=1200)

    if params['do_predict']:
        pass
