## Awesome CNNs

This repo contains a collection of CNN network structures. 

For each kind of CNN network, we provide a pytorch implementation, a caffe implementation for NetScope visualization, and additional pictures(mostly from original paper) to help you understand how it works. Links to original paper and my notes(in Chinese) are also provided.

Furthermore, in experiments folder, there is a training script for you to train each kind of network with Pytorch on dataset CIFAR-10.

### Usage

1.Clone this repo.

```
git clone https://github.com/ddlee96/awesome_cnn.git
```

2.Install PyTorch and TorchVision following instructions on [PyTorch](https://pytorch.org) website.

3.Prepare dataset

```
cd awesome_cnn
mkdir data
```
Then just run `python experiments/experiment.py --data cifar`, it will create `data/cifar/` and download cifar-10 dataset for you. 

(Optional)You can also use dataset MNIST and [Fashion-MNIST](), it requires Pillow package. Install it using `pip install Pillow`.

4.Start Training

```
python experiments/experiment.py --model resnet50 --data cifar --batch_size 182
```
It will train model ResNet50 on cifar10 with batch size 128. More options can be find in `experiment.py`.


### CNN Network List

Network Name|PaperNotes|Paper|Netscope|Proposed Date
--------|--------------------------------------|--------------|---------------|-------
[MobileNets](https://github.com/ddlee96/awesome_cnn/blob/master/models/mobilenet/)| [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://blog.ddlee.cn/2018/01/04/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-MobileNets-Efficient-Convolutional-Neural-Networks-for-Mobile-Vision-Applications/)| [arxiv](https://arxiv.org/abs/1704.04861) | - | 17.04
[ResNeXt](https://github.com/ddlee96/awesome_cnn/blob/master/models/resnext/)| [Aggregated Residual Transformations for Deep Neural Networks](https://blog.ddlee.cn/2018/01/06/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Aggregated-Residual-Transformations-for-Deep-Neural-Networks/)| [arxiv](https://arxiv.org/abs/1611.05431)| [Visualization](http://ethereon.github.io/netscope/#/gist/c2ba521fcb60520abb0b0da0e9c0f2ef) | 16.11
[Xception](https://github.com/ddlee96/awesome_cnn/blob/master/models/xception/) |[Xception: Deep Learning with Depthwise Seperable Convolutions](https://blog.ddlee.cn/2018/01/02/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Xception-Deep-Learning-with-Depthwise-Seperable-Convolutions/) |[arxiv](https://arxiv.org/abs/1610.02357) | [Visualization](http://ethereon.github.io/netscope/#gist/931d7c91b22109f83bbbb7ff1a215f5f) | 16.10
[DenseNet](https://github.com/ddlee96/awesome_cnn/blob/master/models/densenet/)| [Densely Connected Convolutional Networks](https://blog.ddlee.cn/2018/01/06/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Densely-Connected-Convolutional-Networks/)| [arxiv](https://arxiv.org/abs/1608.06993)| [Visualization](http://ethereon.github.io/netscope/#/gist/56cb18697f42eb0374d933446f45b151) | 16.08
[ResNet-v2](https://github.com/ddlee96/awesome_cnn/blob/master/models/resnet_v2/) | Identity Mappings in Deep Residual Networks | [arxiv](https://arxiv.org/abs/1603.05027) | - | 16.03
[Inception-ResNet]((https://github.com/ddlee96/awesome_cnn/blob/master/models/inception-v4/)) |[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://blog.ddlee.cn/2017/12/26/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Inception-v4-Inception-ResNet-and-the-Impact-of-Residual-Connections-on-Learning/) | [arxiv](https://arxiv.org/abs/1602.07261) | [Visualization](http://ethereon.github.io/netscope/#gist/e0ac64013b167844053184d97b380978) | 16.02
[SqueezeNet](https://github.com/ddlee96/awesome_cnn/blob/master/models/squeezenet/)| SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
 |[arxiv](https://arxiv.org/abs/1602.07360)|-| 16.02
[Inception-v3](https://github.com/ddlee96/awesome_cnn/blob/master/models/inception/) |[Rethinking the Inception Architecture for Computer Vision](https://blog.ddlee.cn/2017/12/16/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Rethinking-the-Inception-Architecture-for-Computer-Vision/) | [arxiv](https://arxiv.org/abs/1512.00567) |[Visualization](http://ethereon.github.io/netscope/#gist/a2394c1c4a9738469078f096a8979346) | 15.12
[ResNet](https://github.com/ddlee96/awesome_cnn/blob/master/models/resnet/) | [Deep Residual Learning for Image Recognition](https://blog.ddlee.cn/2017/04/30/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Deep-Residual-Learning-for-Image-Recognition/) | [arxiv](http://arxiv.org/abs/1512.03385) | [Visualization](http://ethereon.github.io/netscope/#/gist/fd5ab897623abec94c4027731ce4e80f) | 15.12
[GoogLeNet](https://github.com/ddlee96/awesome_cnn/blob/master/models/googlenet/) | [Going deeper with convolutions](https://blog.ddlee.cn/2017/11/30/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Going-deeper-with-convolutions/) | [arxiv](https://arxiv.org/abs/1409.4842) | [Visualization](http://ethereon.github.io/netscope/#/gist/db8754ee4b239920b3df5ab93220a84b) | 14.09
[VGG](https://github.com/ddlee96/awesome_cnn/blob/master/models/vgg/)| Very Deep Convolutional Networks for Large-Scale Image Recognition |[arxiv](https://arxiv.org/abs/1409.1556)|[Visualization](http://ethereon.github.io/netscope/#/gist/f671dfd1c382b4b86c9fed14021b4764) | 14.09
[AlexNet](https://github.com/ddlee96/awesome_cnn/blob/master/models/alexnet/) | ImageNet Classification with Deep ConvolutionalNeural Networks |[NIPS](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | [Visualization](http://ethereon.github.io/netscope/#/gist/7c508f9dfa679ee9be936f8fe16b9715) | -
[LeNet](https://github.com/ddlee96/awesome_cnn/blob/master/models/lenet/) | Gradient-based learning applied to document recognition | [Lecun's Homepage](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | - | 1998

### To be added
- [x] SqueezeNet
- [x] ResNet-v2
- [ ] SENet
- [x] LeNet
- [ ] DPN
- [ ] NASNet
- [ ] MobileNet V2