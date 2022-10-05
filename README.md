[license-image]: https://img.shields.io/badge/license-Anti%20996-blue.svg
[license-url]: https://github.com/996icu/996.ICU/blob/master/LICENSE
[pypi-image]: https://badge.fury.io/py/tfts.svg
[pypi-url]: https://pypi.python.org/pypi/tfts
[build-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/time-series-prediction/badge/?version=latest
[docs-url]: https://time-series-prediction.readthedocs.io/en/latest/

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="490" align=center/>
</h1><br>

**[Documentation](https://time-series-prediction.readthedocs.io)** | **[Tutorials](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](https://github.com/LongxingTan/Time-series-prediction/blob/master/README_CN.md)**

**TFTS** (TensorFlow Time Series) is a python package for time series task, supporting the common deep learning methods in TensorFlow.
- Flexible and powerful design for time series task
- Advanced deep learning models
- tfts documentation lives at [time-series-prediction.readthedocs.io](https://time-series-prediction.readthedocs.io)


## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC)

**install**
``` bash
$ pip install tensorflow>=2.0.0
$ pip install tfts
```

**usage**
``` python
import tensorflow as tf
import tfts
from tfts import AutoModel, KerasTrainer

train, valid = tfts.load_data('sine')
backbone = AutoModel('seq2seq')
model = functools.partial(backbone.build_model, input_shape=[24, 2])

trainer = KerasTrainer(model)
trainer.train(train, valid)
trainer.predict(valid[0])
```


## Examples

- [TFTS-Bert model](https://github.com/LongxingTan/KDDCup2022-Baidu) wins the **3rd place** in KDD Cup 2022 Baidu-wind power forecasting
- [TFTS-Seq2seq model](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) wins the **4th place** in Alibaba Tianchi-ENSO prediction 2021

More models
- [RNN](examples/run_prediction.py)
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
- [Parameters tuning by optuna](examples/run_optuna_tune.py)
- [Serving by tf-serving](./examples)

## Performance
The performance is evaluated by this lib's implementation, not official

| Performance | [web traffic<sup>mape</sup>]() | [grocery sales<sup>nwrmsle</sup>](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) | [m5 sales<sub>50</sub><sup>val</sup>]() | [ventilator<sub>75</sub><sup>val</sup>]() |
| :-- | :-: | :-: | :-: | :-: |
| [RNN]() | 672 | 47.7% |52.6% | 61.4% |
| [Seq2seq]() | 672 | 47.7% |52.6% | 61.4% |
| [TCN]() | 672 | 47.7% |52.6% | 61.4% |
| [WaveNet]() | 672 | 47.7% |52.6% | 61.4% |
| [Bert]() | 672 | 47.7% |52.6% | 61.4% |
| [Transformer]() | 672 | 47.7% |52.6% | 61.4% |
| [Informer]() | 672 | 47.7% |52.6% | 61.4% |
| [AutoFormer]() | 672 | 47.7% |52.6% | 61.4% |
| [n-beats]() | 672 | 47.7% |52.6% | 61.4% |
| [U-Net]() | 672 | 47.7% |52.6% | 61.4% |


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
