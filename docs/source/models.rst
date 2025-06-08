Models
======

.. _models:

.. currentmodule:: tfts

Some experiments of tfts in Kaggle Dataset


Models supported
------------------

You can use below models with ``AutoModel``

* RNN
* Seq2seq
* TCN
* WaveNet
* Bert
* Transformer
* DLinear
* NBeats
* AutoFormer
* Informer

.. code-block:: python

    config = AutoConfig.for_model("seq2seq")
    model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)


Add a custom head for tfts model

.. code-block:: python

    config = AutoConfig.for_model("seq2seq")
    model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
    model.project = tf.keras.Sequential(
        layers=[],
        trainable=True,
        name=None
    )
