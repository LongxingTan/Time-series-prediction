# Time series prediction

The repository implement the time series prediction for single variable or multi variables with different models

## Model
- Weibull regression
- LSTM
- WTTE (what time to the next event)
- seq2seq

## Feature
- Kaplan-meier
- Auto regression
- auto correlation
- Time monitoring related features

## Dependencies
- Python 3.6
- Tensorflow 1.12.0
- sklearn 0.20.2
- stat 0.9.0

## Usage
- python run_prediction.py

## Purpose
- The project is to predict the warranty claims' future development, so that we can sense the series issues in advance, as well as measure the vehicle's quality performance in the field for automitive industry

## seq2seq
![model](https://github.com/LongxingTan/Time_series_prediction/blob/master/image/seq2seq.png)