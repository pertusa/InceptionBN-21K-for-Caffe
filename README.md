# Inception-BN full for Caffe

Inception-BN ImageNet (21K classes) model for Caffe.

The model can be downloaded from: http://www.dlsi.ua.es/~pertusa/deep/Inception21k.caffemodel

It was directly converted from the MXNet ImageNet21k model at: https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-21k-inception.md

MXNet Batch Normalization is translated into Caffe using a BatchNorm layer
with the learned mean and variance (and scale=1), followed by a Scale layer that applies the learned gamma and beta.

The file deploy.prototxt was generated with the code at symbol_inception-bn-full.cc.

The code for model conversion (MXNet -> Caffe) can be found here: https://github.com/pertusa/MXNetToCaffeConverter

Licensed under CC0
