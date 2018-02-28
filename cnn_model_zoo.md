# awesome-deepnet-shared-weights

## Table of Contents
- [By Model](#by-model)
- - [VGG](#vgg)
- - [ResNet](#resnet)
- [By Framework](#by-framework
)

## By Model
### VGG
#### Paper
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
#### Weights
Framework|Info|Notes
-----------------|------------------------|---------------------
[Keras](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1)|[info](https://github.com/fchollet/deep-learning-models#extract-features-from-images)| hosted by Github
[Caffe, VGG16](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)|[info](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md), [config](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt)| directly
[Caffe, VGG19](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel)|[info](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md), [config](https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt)| directly
[mxnet, VGG16](http://data.dmlc.ml/mxnet/models/imagenet/vgg/vgg16.tar.gz)|[info](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-vgg.md)| directly
[mxnet, VGG19](http://data.dmlc.ml/mxnet/models/imagenet/vgg/vgg19.tar.gz)|[info](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-vgg.md)| directly
[MatConvNet, VGG16](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat)|[info](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification)| directly
[MatConvNet, VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)|[info](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification)| directly
[Tensorflow, VGG16](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)|[blog](https://www.cs.toronto.edu/~frossard/post/vgg16/)| directly
[Tensorflow-slim, VGG16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)| [Instructions](https://github.com/tensorflow/models/tree/master/slim#fine-tuning-a-model-from-an-existing-checkpoint) | directly
[Tensorflow-slim, VGG19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)| [Instructions](https://github.com/tensorflow/models/tree/master/slim#fine-tuning-a-model-from-an-existing-checkpoint) | directly

### ResNet
#### Paper
[Deep Residual Learning for Image Recognition ](https://arxiv.org/abs/1512.03385)
#### Weights
Framework|Info|Notes
-----------------|------------------------|-------------------
[Keras](https://github.com/fchollet/deep-learning-models/releases/tag/v0.2)|[info](https://github.com/fchollet/deep-learning-models#classify-images)| hosted by Github
[Caffe](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)|[info](https://github.com/KaimingHe/deep-residual-networks#models)| hosted by OneDrive
[Tensorflow](https://raw.githubusercontent.com/ry/tensorflow-resnet/master/data/tensorflow-resnet-pretrained-20160509.tar.gz.torrent)|[info](https://github.com/ry/tensorflow-resnet#resnet-in-tensorflow)| Torrent
[MatConvNet](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat)|[info](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification)| directly
[Torch](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained#trained-resnet-torch-models)|[blog](http://torch.ch/blog/2016/02/04/resnets.html), [Instruction](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained#fine-tuning-on-a-custom-dataset)| hosted by cloudfront
[Tensorflow-slim](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)| [Instructions](https://github.com/tensorflow/models/tree/master/slim#fine-tuning-a-model-from-an-existing-checkpoint) |

## By Framework
Framework | Repo
---------------|--------------
Keras|[fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models)
Caffe|[Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
mxnet|[Model Gallery](https://github.com/dmlc/mxnet-model-gallery)
MatConvNet|[Pretrained models](http://www.vlfeat.org/matconvnet/pretrained/)
Tensorflow-slim|[Pre-trained Models](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)
