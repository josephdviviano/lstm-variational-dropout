Variational Dropout for LSTMs
-----------------------------

A simple implementation of

A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
Yarin Gal & Zoubin Ghahramani, 2016. [arxiv](https://arxiv.org/abs/1512.05287)

This implementation applies dropout to the activations of the LSTM rather than
dropping the weights as it was easier to apply this strategy within Pytorch's
framework. This is in contrast to previous implementations (e.g., WeightDrop
from [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) and
[torchNLP](https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html)).

