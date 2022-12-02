Quick start
===============

.. _quick-start:

.. _installation:

Installation
--------------

Install `time series tfts <https://github.com/LongxingTan/Time-series-prediction>`_, follow the installation instructions first

* Python 3.7+
* `TensorFlow 2.x <https://www.tensorflow.org/install/pip>`_ installation instructions

Now you are ready, proceed with

.. code-block:: shell

    $ pip install tfts


You can run it in docker, download the Dockerfile to host server

.. code-block:: shell

    $ docker build -f ./Dockerfile -t "custom image name" .
    $ docker run --rm -it --init --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "custom image name" /bin/bash

.. _usage:

Usage
-------------

.. currentmodule:: tfts

The general setup for training and testing a model is

#. Build time series 3D training dataset and valid dataset. The shape of input and label are (examples, train_sequence_length, features) and (examples, predict_sequence_length, 1)
#. Instantiate a model using the the ``AutoModel()`` method
#. Create a ``Trainer()`` or ``KerasTrainer()`` object. Define the optimizer and loss function in trainer
#. Train the model on the training dataset and check if it has converged with acceptable accuracy
#. Tune the hyper-parameters of the model and training
#. Load the model from the model checkpoint and apply it to new data

If you prefer other DL frameworks, try `pytorch-forecasting <https://github.com/jdb78/pytorch-forecasting>`_, `gluonts <https://github.com/awslabs/gluonts>`_, `paddlets <https://github.com/PaddlePaddle/PaddleTS>`_

Example
~~~~~~~~~~~~~

.. code-block:: python

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import tfts
    from tfts import AutoModel, KerasTrainer

    # load data
    train_length = 36
    predict_length = 12
    train, valid = tfts.load_data('sine', train_length, predict_length)

    # build model
    model = AutoModel('seq2seq', predict_length=predict_length)

    # train
    opt = tf.keras.optimizers.Adam(0.003)
    loss_fn = tf.keras.losses.MeanSquaredError()
    trainer = KerasTrainer(model, loss_fn=loss_fn, optimizer=opt)
    trainer.train(train, valid, n_epochs=10, batch_size=32)

    # test
    trainer.predict(valid[0])


.. currentmodule:: tfts
