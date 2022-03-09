# Wavenet

## Introduction
Wavenet is introduced in [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) by DeepMind, first used in audio generation. Its main components use the causal dilated convolutional neutral network. The kernel of CNN layer share the same weights, so it can also be used to percept the seasonality of time series issue.

The dilated causal convolutional layer <br>
![wavenet](./assets/wavenet.gif)

It's become a promising method in time series since sjv open-sourced his repo: [web-traffic-forecasting](https://github.com/sjvasquez/web-traffic-forecasting)

## Some detail
#### casual dilated convolutional neutral network
Casual: make sure that the future information won't leak

Dilated: extend the receptive field to track the long term dependencies
Implementation:

Normal convolution:

### Encoder
encoder inputs: a 3D sequence data, shape: [batch_size, train_sequence_length, n_encoder_feature]

dense layer: for each point in sequence , project the n_encoder_feature to  n_residual_channel, and each point share the same weights.

deep conv layer: for each conv layer, the dilation rate is twice than last to expand the receptive field. And each conv layer has 2* n_residual_channel filters and fixed 2 kernel size. The block of conv layer has time_conv, gate, and another dense. Then the output is splitted into two part, one as the residual, the other as interim result. The residual will be added with input and continue as input for next layer.

postprocess: concat all the interim result together, and add the activation and dense layer to proejct the last feature dimension to 1.

outputs: the postprocess result, and the interim result.
the postprocess result is a tensor with shape [batch_size, train_sequence_length, n_dense_dim]
the interim result is a list with length of n_conv_layers, each item is a tensor with shape [batch_size, train_sequence_length, n_conv_filters]

### Decoder
decoder inputs: 3D decoder features, shape: [batch_size, predict_sequence_length, n_decoder_feature]
encoder interim result:

the basic step in decoder is a loop to predict each step:
But it's hard to implement a recursive version, unless it's only uni-variable prediction, the history of target is used only as its feature.

So for complex features, the interim result from encoder can be included.
Each layer, we choose the last n_dilation vector from the last layer's sequence.


Here i also plan to use the trick from transformer: decoder mask to do it.


## Performance

```bash
cd examples
python run_train.py --use_model wavenet
```


## Further reading
[seriesnet: unique variable time series prediction](https://github.com/kristpapadopoulos/seriesnet/blob/master/seriesnet.py)<br>
[wavenet: multi-variables time series](https://github.com/sjvasquez/web-traffic-forecasting/blob/master/cnn.py)<br>

