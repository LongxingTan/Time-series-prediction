Getting started
===============

.. _getting-started:


Installation
--------------

.. _install:

If you are working windows, you need to first install TensorFlow with

``pip install tensorflow``.

Otherwise, you can proceed with

``pip install tfts``



Usage
-------------

.. currentmodule:: tfts

The general setup for training and testing a model is

#. Create training dataset using :py:class:`~data.timeseries.TimeSeriesDataSet`.
#. Using the training dataset, create a validation dataset with :py:meth:`~data.timeseries.TimeSeriesDataSet.from_dataset`.
   Similarly, a test dataset or later a dataset for inference can be created. You can store the dataset parameters
   directly if you do not wish to load the entire training dataset at inference time.

#. Instantiate a model using the its ``.from_dataset()`` method.
#. Create a ``pytorch_lightning.Trainer()`` object.
#. Find the optimal learning rate with its ``.tuner.lr_find()`` method.
#. Train the model with early stopping on the training dataset and use the tensorboard logs
   to understand if it has converged with acceptable accuracy.
#. Tune the hyperparameters of the model with your
#. Train the model with the same learning rate schedule on the entire dataset.
#. Load the model from the model checkpoint and apply it to new data.


The :ref:`Tutorials <tutorials>` section provides detailled guidance and examples on how to use models and implement new ones.


Example
--------


.. code-block:: python

    import tensorflow as tf
    from tfts import Trainer, build_tfts_model

    # load data
    data = ...

    # define dataset
    max_encoder_length = 36
    max_prediction_length = 6
    training_cutoff = "YYYY-MM-DD"  # day for cutoff

    training = TimeSeriesDataSet(
        data[lambda x: x.date < training_cutoff],
        time_idx= ...,
        target= ...,
        # weight="weight",
        group_ids=[ ... ],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[ ... ],
        static_reals=[ ... ],
        time_varying_known_categoricals=[ ... ],
        time_varying_known_reals=[ ... ],
        time_varying_unknown_categoricals=[ ... ],
        time_varying_unknown_reals=[ ... ],
    )

    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    # define trainer with early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=0,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[lr_logger, early_stop_callback],
    )

    # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = trainer.tuner.lr_find(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    # fit the model
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

Main API
---------

.. currentmodule:: tfts
.. moduleautosummary::
   :toctree: api
   :template: custom-module-template.rst
   :recursive:

   tfts
