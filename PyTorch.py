# -*- coding:utf-8 -*-
# @FileName :PyTorch.py
# @Time :2023/4/12 14:26
# @Author :Xiaofeng
import torch
import torch.nn as nn
import torch.optim as optim
from torch import flatten, device
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F  # useful stateless functions
import numpy as np
from tqdm import tqdm

USE_GPU = True
dtype = torch.float32  # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    # def preparation():


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
    x = torch.zeros((64, 50), dtype=dtype)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
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


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        # conv weight [out_channel, in_channel, kH, kW]
        # np.prod()计算所有元素的乘积
        fan_in = np.prod(shape[1:])
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


def initialize_weight():
    # create a weight of shape [3 x 5]
    # you should see the type `torch.cuda.FloatTensor` if you use GPU.
    # Otherwise it should be `torch.FloatTensor`
    print(random_weight((3, 5, 4)))
    print(random_weight((3, 5, 4)).type())


def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.

    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model

    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.

    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD

    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # 反向传播：PyTorch计算出计算图中哪些张量需要_grad=True
        # 使用反向传播计算相对于这些张量的损耗_Loss及梯度_grad，并将梯度存储在每个张量的.grad属性中。
        loss.backward()

        # 更新参数。我们不想通过参数更新进行反向传播
        # 所以我们在torch.no_grad()#上下文管理器下确定更新范围，以防止生成计算图。
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # 运行反向传播后手动调零梯度
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()


def test_train_part2():
    hidden_layer_size = 4000
    learning_rate = 1e-2

    w1 = random_weight((3 * 32 * 32, hidden_layer_size))
    w2 = random_weight((hidden_layer_size, 10))

    train_part2(two_layer_fc, [w1, w2], learning_rate)


def test_train_convNet():
    learning_rate = 3e-3

    channel_1 = 32
    channel_2 = 16

    ################################################################################
    # TODO: Initialize the parameters of a three-layer ConvNet.                    #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    conv_w1 = random_weight((channel_1, 3, 5, 5))  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = zero_weight((channel_1,))
    conv_w2 = random_weight((channel_2, channel_1, 3, 3))
    conv_b2 = zero_weight((channel_2,))
    fc_w = random_weight((channel_2 * 32 * 32, 10))
    fc_b = zero_weight(10)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    train_part2(three_layer_convnet, params, learning_rate)


"""
Barebone PyTorch要求我们手动跟踪所有参数张量。这对于只有几个张量的小型网络来说是可以的
但在较大的网络中跟踪几十或几百个张量会非常不方便，而且容易出错。

PyTorch为您提供nn.Module API来定义任意的网络架构，同时为您跟踪每一个可学习的参数。
在第二部分中，我们自己实现了SGD。PyTorch还提供了torch.optim包，该包实现了所有常见的优化器，如RMSProp、Adagrad和Adam。
它甚至支持近似二阶方法，如L-BFGS！您可以参考文档以了解每个优化器的确切规格
"""


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        在构造函数__init__()中，将所需的所有层定义为类属性。
        像 nn.Lineral和nn.Conv2d这样的层对象本身就是nn.Module子类，并包含可学习的参数，因此您不必自己实例化原始张量。
        模块将为您跟踪这些内部参数。请参阅文档以了解有关数十个内置层的更多信息。warn：不要忘记调用super()__先初始化()
        Args:
            input_size:
            hidden_size:
            num_classes:
        """
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # forward always defines connectivity
        # 在forward()方法中，定义网络的连接。您应该使用__init__中定义的属性作为函数调用
        # 将张量作为输入并输出“转换”的张量。不要在forward()中创建任何具有可学习参数的新层
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores


def test_TwoLayerFC():
    input_size = 50
    x = torch.zeros((64, input_size), dtype=dtype)  # minibatch size 64, feature dimension 50
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=channel_1, kernel_size=5, padding=2, bias=True)
        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias, 0)

        self.relu = F.relu

        self.conv_2 = nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias, 0)

        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        conv_1 = self.relu(self.conv_1(x))
        conv_2 = self.relu(self.conv_2(conv_1))
        scores = self.fc(flatten(conv_2))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores


def test_ThreeLayerConvNet():
    learning_rate = 1e-2
    # x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    train_part34(model, optimizer)


def check_accuracy_part34(loader, model):
    # if loader.dataset.train:
    #     print('Checking accuracy on validation set')
    # else:
    #     print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return num_correct, num_samples, acc
        # print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    best_acc = 0.0
    best_model = None
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                num_correct, num_samples, acc = check_accuracy_part34(loader_val, model)
                print('epoch %d / %d , Iteration %d, loss = %.4f , Got %d / %d correct (%.2f)' % (
                    e, epochs, t, loss.item(), num_correct, num_samples, 100 * acc))
                if 100 * acc > best_acc:
                    best_acc = 100 * acc
                    best_model = model
    return best_acc, best_model


def Train_Two_Layer_Network():
    hidden_layer_size = 4000
    learning_rate = 1e-2
    model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_part34(model, optimizer)


def Train_Three_Layer_Network():
    """
    You should now use the Module API to train a three-layer ConvNet on CIFAR.
    This should look very similar to training the two-layer network!
    You don't need to tune any hyperparameters, but you should achieve above above 45% after training for one epoch.
    You should train the model using stochastic gradient descent without momentum.
    """
    learning_rate = 3e-3
    channel_1 = 32
    channel_2 = 16

    model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_part34(model, optimizer)


def sequential_nn_Two_Layer():
    # We need to wrap `flatten` function in a module in order to stack it
    # in nn.Sequential
    class Flatten(nn.Module):
        def forward(self, x):
            return flatten(x)

    hidden_layer_size = 4000
    learning_rate = 1e-2

    model = nn.Sequential(
        Flatten(),
        nn.Linear(3 * 32 * 32, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, 10),
    )

    # you can use Nesterov momentum in optim.SGD
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    train_part34(model, optimizer)


def sequential_Three_layer_convNet():
    channel_1 = 96
    channel_2 = 80
    channel_3 = 54
    channel_4 = 48
    channel_5 = 32
    channel_6 = 20
    channel_7 = 16
    learning_rate = 1e-2

    class Flatten(nn.Module):
        def forward(self, x):
            return flatten(x)

    model = nn.Sequential(

        nn.Conv2d(in_channels=3, out_channels=channel_1, kernel_size=5, padding=2, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(channel_1),
        nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=3, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(channel_2),

        nn.Conv2d(in_channels=channel_2, out_channels=channel_3, kernel_size=3, padding=1, stride=2, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(channel_3),
        nn.Conv2d(in_channels=channel_3, out_channels=channel_4, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(channel_4),

        nn.Conv2d(in_channels=channel_4, out_channels=channel_5, kernel_size=2, padding=1, stride=2, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(channel_5),
        nn.Conv2d(in_channels=channel_5, out_channels=channel_6, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(channel_6),

        nn.Conv2d(in_channels=channel_6, out_channels=channel_7, kernel_size=2, padding=1, stride=1, bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, padding=1, stride=2),

        Flatten(),
        nn.Linear(channel_7 * 6 * 6, 12 * 5 * 5),
        nn.Linear(12 * 5 * 5, 10)
    )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    best_acc, best_model = train_part34(model, optimizer, epochs=10)
    print('best_accuracy: %.2f' % best_acc)

    num_correct, num_samples, acc = check_accuracy_part34(loader_test, best_model)
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


if __name__ == '__main__':
    sequential_Three_layer_convNet()
