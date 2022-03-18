![Time series prediction](./docs/source/_static/logo.svg)

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[build-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml/badge.svg?branch=master

**[Documentation](https://time-series-prediction.readthedocs.io)** | **[Tutorials](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](https://github.com/LongxingTan/Time-series-prediction/blob/master/README_CN.md)**

TFTS (TensorFlow Time Series) is a python package for time series prediction, supporting the common deep learning methods in TensorFlow. 


## Usage

1. Install the required library
```bash
$ pip install -r requirements.txt
```
2. Download the data, if necessary
```bash
$ sh ./data/download_passenger.sh
```
3. Train the model <br>
set `custom_model_params` if you want (refer to params in `./tfts/models/*.py`), and pay attention to feature engineering.

```bash
$ cd examples
$ python run_train.py --use_model seq2seq
$ cd ..
$ tensorboard --logdir=./data/logs
```
4. Predict new data
```bash
$ cd examples
$ python run_test.py
```

## Documentation
Visit [https]() to read the documentation with detailed tutorials

## Examples
- I use the seq2seq model from this lib to win 4th/2849 in Tianchi ENSO prediction, code is [here](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction)

<!--
| Model | [Web Traffic<sup>mape</sup>]() | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [seq2seq]() | 672 | 47.7% |52.6% | 61.4% | 
| [deepar]() | 672 | 47.7% |52.6% | 61.4% | 
| [wavenet]() | 672 | 47.7% |52.6% | 61.4% | 
| [tcn]() | 672 | 47.7% |52.6% | 61.4% | 
| [transformer]() | 672 | 47.7% |52.6% | 61.4% | 
| [transformer]() | 672 | 47.7% |52.6% | 61.4% | 
| [n-beats]() | 672 | 47.7% |52.6% | 61.4% | 
| [u-net]() | 672 | 47.7% |52.6% | 61.4% |
|  |  |  |  |  |
Please Note that: the performance above is only representing this repo's current implementation performance.
-->

## Further reading
- https://github.com/awslabs/gluon-ts/
- https://github.com/Azure/DeepLearningForTimeSeriesForecasting
- https://github.com/microsoft/forecasting
- https://github.com/jdb78/pytorch-forecasting
- https://github.com/timeseriesAI/tsai
- https://github.com/facebookresearch/Kats
