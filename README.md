<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="500" align=center/>
</h1><br>

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

**[Documentation](https://time-series-prediction.readthedocs.io)** | **[Tutorials](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](https://github.com/LongxingTan/Time-series-prediction/blob/master/README_CN.md)**

**TFTS** (TensorFlow Time Series) is a python package for time series task, supporting the common deep learning methods on TensorFlow.
- Flexible and powerful design for time series task
- [Advanced deep learning models](./examples)
- [Detailed documentation with tutorials](https://time-series-prediction.readthedocs.io)


### Install
``` bash
pip install tensorflow>=2.0.0
pip install tfts
```


### Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC)

``` python
import tensorflow as tf
import tfts

data = tfts.load('passenger')
x_train, y_train
model = AutoModel('seq2seq')
# train the model
model.train(data)
# predict new data
model.predict(data)

```


## Examples
Highlights
- [TFTS-Bert model](https://github.com/LongxingTan/KDDCup2022-Baidu) wins the 3rd place in KDD Cup 2022 Baidu-wind power forecasting
- [TFTS-Seq2seq model](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) wins the 4th place in Alibaba Tianchi-ENSO prediction 2021

More models 
- [RNN](./examples/run_rnn.py) 
- [DeepAR](./examples/run_deepar.py) 
- [ESRNN](./examples/run_esrnn.py) 
- [Seq2seq](./examples/run_seq2seq.py)
- [TCN](./examples/run_tcn.py)
- [Wavenet](./examples/run_wavenet.py)
- [Bert](./examples/run_bert.py)
- [Transformer](./examples/run_transformer.py)
- [Temporal-fusion-transformer](./examples/run_temporal_fusion_transformer.py)
- [Informer](./examples/run_informer.py)
- [AutoFormer](./examples/run_autoformer.py)
- [U-Net](./examples/run_unet.py)
- [Nbeats](./examples/run_nbeats.py)
- [GAN](./examples/run_gan.py)

More demos 
- [Time series classification](./examples/run_classification.py)
- [Anomaly detection](./examples/run_anomaly.py)
- [Uncertainty prediction](./examples/run_uncertrainty.py)
- [Parameters tuning with optuna](./examples/run_optuna.py)
- [tf-serving](./examples/)


## Citation

Please use the following reference to cite our tfts library 
```
@misc{tfts2020,
  author = {Longxing Tan},
  title = {Time series prediction},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/longxingtan/time-series-prediction}},
}
```
