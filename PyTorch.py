# -*- coding:utf-8 -*-
# @FileName :PyTorch.py
# @Time :2023/4/12 14:26
# @Author :Xiaofeng
import torch
import torch.nn as nn
import torch.optim as optim
from torch import flatten
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F  # useful stateless functions
import numpy as np


def gpu():
    # You can manually switch to a GPU device on Colab by clicking Runtime -> Change runtime type and selecting GPU
    # under Hardware Accelerator. You should do this before running the following cells to import packages,
    # since the kernel gets restarted upon switching runtimes.
    USE_GPU = True
    dtype = torch.float32  # We will be using float throughout this tutorial.

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Constant to control how frequently we print train loss.
    print_every = 100
    print('using device:', device)


def preparation():
    """
    The torchvision.transforms package provides tools for preprocessing data
    and for performing data augmentation; here we set up a transform to
    preprocess the data by subtracting the mean RGB value and dividing by the
    standard deviation of each RGB value; we've hardcoded the mean and std.

    """
    NUM_TRAIN = 49000

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # We set up a Dataset object for each split (train / val / test); Datasets load
    # training examples one at a time, so we wrap each Dataset in a DataLoader which
    # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
    # training set into train and val sets by passing a Sampler object to the
    # DataLoader telling how it should sample from the underlying Dataset.
    cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                                 transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64,
                              sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                               transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                                transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)


def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass;
    PyTorch will take care of the backward pass for us.

    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).

    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]

    w1, w2 = params

    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    # you can also use `.clamp(min=0)`, equivalent to F.relu()
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x


def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros(64, 50)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros(50, hidden_layer_size)
    w2 = torch.zeros(hidden_layer_size, 10)
    scores = two_layer_fc(x, [w1, w2])
    # print(x.type())
    print(scores.size())  # you should see [64, 10]


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    channel_1, _, KH1, KW1 = conv_w1.shape
    channel_2, _, KH2, KW2 = conv_w2.shape
    conv_1 = F.conv2d(x, conv_w1, bias=conv_b1, padding=2)
    relu_1 = F.relu(conv_1)
    conv_2 = F.conv2d(relu_1, conv_w2, bias=conv_b2, padding=1)
    relu_2 = F.relu(conv_2)
    scores = flatten(relu_2).mm(fc_w) + fc_b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores


def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32))  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5))  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3))  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]


if __name__ == '__main__':
    # gpu()
    # test_flatten()
    # two_layer_fc_test()
    three_layer_convnet_test()
