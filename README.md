# Time series prediction
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)<br>
This repository implements the common methods of time series prediction, especially deep learning methods in TensorFlow2. 
It's welcomed to contribute if you have any better idea, just create a PR. If any question, feel free to open an issue.

#### Ongoing project, I will continue to improve this, so you might want to watch/star this repo to revisit.

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

## Usage
1. Install the required library
```bash
$ pip install -r requirements.txt
```
2. Download the data, if necessary
```bash
$ bash ./data/download_passenger.sh
```
3. Train the model <br>
set `custom_model_params` if you want (refer to params in `./deepts/models/*.py`), and pay attention to feature engineering.

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

## Further reading
- https://github.com/awslabs/gluon-ts/
- https://github.com/Azure/DeepLearningForTimeSeriesForecasting
- https://github.com/microsoft/forecasting
- https://github.com/jdb78/pytorch-forecasting
- https://github.com/timeseriesAI/tsai
