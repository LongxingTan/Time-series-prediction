Quick start
===============

.. _quick-start:

.. _install:

Installation
--------------

Install `time series tfts <https://github.com/LongxingTan/Time-series-prediction>`_, follow the installation instructions first

* Python 3.7+
* `TensorFlow 2.0 <https://www.tensorflow.org/install/pip>`_ installation instructions

Now you are ready, proceed with

.. code-block:: shell

    $ pip install tfts


You can also run it in docker

.. code-block:: shell

    $ docker run

Usage
-------------

.. currentmodule:: tfts

The general setup for training and testing a model is

#. Build time series 3D training dataset and valid dataset. The shape of input and label are (examples, train_sequence_length, features) and (examples, predict_sequence_length, 1)
#. Instantiate a model using the the ``AutoModel()`` method
#. Create a ``Trainer()`` or ``KerasTrainer()`` object. Define the optimizer and loss function in trainer
#. Train the model on the training dataset and check if it has converged with acceptable accuracy
#. Tune the hyper-parameters of the model and training
#. Load the model from the model checkpoint and apply it to new data.


Example
--------

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, KerasTrainer

    # load data
    train_length = 36
    predict_length = 12
    train, valid = tfts.load_data('sine', train_length, predict_length)

    # build model
    model = AutoModel('seq2seq', predict_length=predict_length)

    # train
    epochs = 10
    batch_size = 32
    opt = tf.keras.optimizers.Adam(0.003)
    loss_fn = tf.keras.losses.MeanSquaredError()
    trainer = KerasTrainer(model, loss_fn=loss_fn, optimizer=opt)
    trainer.train(train, valid, n_epochs=epochs, batch_size=batch_size)

    # test
    trainer.predict(valid[0])


Main API
---------

.. currentmodule:: tfts
