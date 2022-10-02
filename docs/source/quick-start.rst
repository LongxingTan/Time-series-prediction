Quick start
===============

.. _quick-start:


.. _install:

Installation
--------------

Before you start, you need to first install `TensorFlow2 <https://www.tensorflow.org/install>`_ with

.. code-block:: shell

    $ pip install tensorflow

Then, proceed with

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

    import functools
    import tensorflow as tf
    import tfts
    from tfts import AutoModel, KerasTrainer

    # load data
    train_sequence_length = 36
    predict_sequence_length = 12
    train, valid = tfts.load_data('sine', train_sequence_length, predict_sequence_length)

    # build model
    backbone = AutoModel('seq2seq')
    model = functools.partial(backbone.build_model, input_shape=[train_sequence_length, 2])

    # train
    epochs = 10
    batch_size = 32
    opt = tf.keras.optimizers.Adam(0.003)
    loss_fn = tf.keras.losses.MeanSquaredError()
    trainer = KerasTrainer(model, loss_fn=loss_fn, optimizer=opt)
    trainer.train(train, valid)

    # test
    trainer.predict(valid[0])


Main API
---------

.. currentmodule:: tfts
