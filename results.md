### CIFAR10 PyTorch Test

Transform:
RamdomCrop, RandomHorizontalFlip, Normalize
BatchSize:
128

LeNet ~500M
Epoch: 43 train loss: 0.949, train acc: 0.663
  val loss: 0.889, val acc: 0.691
  time: 10.2s

GoogLeNet ~2100M
Epoch: 15 train loss: 0.177, train acc: 0.938
  val loss: 0.355, val acc: 0.895
  time: 82.0s

ResNet50 ~1800M
Epoch: 17 train loss: 0.233, train acc: 0.920
  val loss: 0.342, val acc: 0.893
  time: 70.8s

VGG16
Epoch: 28 train loss: 0.121, train acc: 0.959
  val loss: 0.350, val acc: 0.900
  time: 27.1s

ResNeXt29_2x64d ~1800M
Epoch: 17 train loss: 0.217, train acc: 0.924
  val loss: 0.368, val acc: 0.887
  time: 56.0s

PreActResNet18 ~1100M
Epoch: 21 train loss: 0.167, train acc: 0.942
  val loss: 0.358, val acc: 0.897
  time: 26.1s

MobileNet ~800M
Epoch: 38 train loss: 0.190, train acc: 0.932
  val loss: 0.387, val acc: 0.883
  time: 20.7s