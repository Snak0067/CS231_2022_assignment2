# -*- coding:utf-8 -*-
# @FileName :FullyConnectedNets.py
# @Time :2023/4/4 12:44
# @Author :Xiaofeng
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

data = get_CIFAR10_data()


def rel_error(x, y):
    """Returns relative error."""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def initial_Loss_and_Gradient_check():
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print("Running check with reg = ", reg)
        model = FullyConnectedNet(
            [H1, H2],
            input_dim=D,
            num_classes=C,
            reg=reg,
            weight_scale=5e-2,
            dtype=np.float64
        )

        loss, grads = model.loss(X, y)
        print("Initial loss: ", loss)

        # Most of the errors should be on the order of e-7 or smaller.
        # NOTE: It is fine however to see an error for W2 on the order of e-5
        # for the check when reg = 0.0
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print(f"{name} relative error: {rel_error(grad_num, grads[name])}")


if __name__ == '__main__':
    initial_Loss_and_Gradient_check()
