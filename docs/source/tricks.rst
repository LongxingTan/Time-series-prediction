Tricks
======

.. _tricks:

Use tfts in competition flexible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want a better performance, you can import tfts source code to modify it directly. That's how I use it in competitions.

* `The TFTS BERT model <https://github.com/LongxingTan/KDDCup2022-Baidu>`_ wins the 3rd place in `Baidu KDD Cup 2022 <https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction>`_
* `The TFTS Seq2Seq mode <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_ wins the 4th place of `Tianchi ENSO prediction <https://tianchi.aliyun.com/competition/entrance/531871/introduction>`_

General Tricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no free launch, and it's impossible to forecast the future. So we should understand first how to forecast based on the trend, seasonality, cyclicity and noise.

* target transformation

	skip connect. skip connect from ResNet is a special and common target transformation, tfts provides some basic skip connect in model config. If you want try more skip connect, please use ``AutoModel`` to make custom model.

* different temporal scale

	we can train different models from different scale


Multi-steps prediction strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* multi models

* add a hidden-sizes dense layer at last

* encoder-decoder structure

* encoder-forecasting structure
