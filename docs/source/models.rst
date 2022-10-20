Models
======

.. _models:

.. currentmodule:: tfts

Some experiments of tfts in Kaggle Dataset

===========  ===========  =============  ========  ==========
Performance  Web traffic  Grocery sales  M5 sales  Ventilator
===========  ===========  =============  ========  ==========
RNN          False        False          False      False
DeepAR       False        True           False      False
Seq2seq      True         True           False      False
TCN          True         True           False      False
===========  ===========  =============  ========  ==========

Models supported
------------------

you can you below models in ``AutoModel()``

* RNN
* DeepAR
* Seq2seq
