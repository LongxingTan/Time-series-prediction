# Time series prediction
The repository implements the time series prediction for single variable or multi variables with different models, with prototypes for experimental purposes only

## Model
- Weibull regression
- ARIMA
- LSTM & WTTE (what time to the next event)
- seq2seq
- GAN
- xgb
- svm

## Feature
- Kaplan-meier (for survival analysis)
- Auto regression
- Auto correlation
- Time monitoring related features

## Dependencies
- Python 3.6
- Tensorflow 1.12.0
- sklearn 0.20.2
- stat 0.9.0

## Usage
<pre><code>
- python run_prediction.py
- python run_dl.py (for deep learning methods, it's been rewriten by tf.data and tf.estimator)
</code></pre>

## Introduction
The project is to predict the warranty claims' future development, so that we can sense the series issues in advance, as well as measure the vehicle's quality performance in the field for automitive industry
1. create data (if necessary)
2. load data (it supportes different data sources)
3. create features (refer to [introduction](https://github.com/LongxingTan/Time_series_prediction/blob/master/create_features_intro.ipynb))
4. prepare model input
5. model
6. run prediction

## Model detail
- seq2seq
![model](https://github.com/LongxingTan/Time_series_prediction/blob/master/image/seq2seq.png)

- ARIMA