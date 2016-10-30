# Wide ResNet implemented with TensorLayer

## Original Paper's Abstract
The paper on Wide Residual Networks (BMVC 2016) [http://arxiv.org/abs/1605.07146](http://arxiv.org/abs/1605.07146) is by Sergey Zagoruyko and Nikos Komodakis.

>Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train.

>To tackle these problems, in this work we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts.

>For example, we demonstrate that even a simple 16-layer-deep wide residual network outperforms in accuracy and efficiency all previous deep residual networks, including thousand-layer-deep networks. We further show that WRNs achieve incredibly good results (e.g., achieving new state-of-the-art results on CIFAR-10, CIFAR-100 and SVHN) and train several times faster than pre-activation ResNets.

The performance over original ResNets is substantial and this should be used over the original ResNets.

## Original Implementation in Lua and Torch
[Lua and Torch Repository Link](https://github.com/szagoruyko/wide-residual-networks)

## Installation Instructions for TensorLayer
```
[stable version] pip install tensorlayer
[master version] pip install git+https://github.com/zsdonghao/tensorlayer.git
```

## TensorLayer Implementation Credits
I would like to give credits to [Hao Dong](https://github.com/zsdonghao/tensorlayer) for helping out with this new package as I was originally unfamiliar with the API. You are able to get up to speed quickly if you have experience with Keras. 

## Files
I have included two files. 

1. cifar_wide_resnet_tl.py
	- This allows you to run each iteration manually with an external call. 
	- It illustrates how TensorLayer allows for more complex use.
	- It runs for 10 iterations, you can easily increase the number of iterations. 
2. cifar_wide_resnet_keras.py
	- This is a similar implementation with Keras.
	- However extending this further is challenging.

