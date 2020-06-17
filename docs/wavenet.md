# Wavenet

## Introduction
Wavenet is introduced in [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) by DeepMind, first used in audio generation. Its main components use the causal dilated convolutional neutral network. The kernel of CNN layer share the same weights, so it can also be used to percept the seasonality of time series issue.

The dilated causal convolutional layer
![wavenet](https://github.com/LongxingTan/Time-series-prediction/blob/master/docs/assets/wavenet.gif)

It's become popular in time series application since sjv open-sourced his repo [web-traffic-forecasting](https://github.com/sjvasquez/web-traffic-forecasting)

## Some detail
#### casual dilated convolutional neutral network
Casual: make sure that the future information won't leak

Dilated: extend the receptive field to track the long term dependencies
Implementation:

Normal convolution:


## Performance


## Further reading

