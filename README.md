# Inception-BN full for Caffe

Inception-BN ImageNet (21K classes) model for Caffe.

Model weights and prototxt were converted from the MXNet ImageNet21k model at: https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-21k-inception.md

There are minor differences in the Caffe and MXNet outputs, probably due to the behaviour of MXNet padding in pooling layers (see https://github.com/dmlc/mxnet/issues/2718).

The trained model can be downloaded from: http://www.dlsi.ua.es/~pertusa/deep/Inception21k.caffemodel

Licensed under CC0
