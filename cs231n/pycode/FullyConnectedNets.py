# -*- coding:utf-8 -*-
# @FileName :FullyConnectedNets.py
# @Time :2023/4/4 12:44
# @Author :Xiaofeng
import time
import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.cnn import ThreeLayerConvNet
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

data = get_CIFAR10_data()
num_train = 50
small_data = {
    "X_train": data["X_train"][:num_train],
    "y_train": data["y_train"][:num_train],
    "X_val": data["X_val"],
    "y_val": data["y_val"],
}


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


def model_tweaking(learning_rate, weight_scale):
    model = FullyConnectedNet(
        [100, 100, 100, 100],
        weight_scale=weight_scale,
        dtype=np.float64
    )
    solver = Solver(
        model,
        small_data,
        print_every=10,
        num_epochs=20,
        batch_size=25,
        update_rule="sgd",
        optim_config={"learning_rate": learning_rate},
    )
    solver.train()
    return solver.train_acc_history, solver.loss_history


def sanity_check():
    # TODO: Use a three-layer Net to overfit 50 training examples by
    # tweaking just the learning rate and initialization scale.

    not_reach = True
    bset_lr, best_wc, loss_history = None, None, None
    while not_reach:
        weight_scale = 10 ** (np.random.uniform(-6, -1))
        learning_rate = 10 ** (np.random.uniform(-4, -1))
        train_acc_history, loss_history = model_tweaking(learning_rate, weight_scale)
        if max(train_acc_history) == 1.0:
            not_reach = False
            best_lr = learning_rate
            best_wc = weight_scale
            print("best learning_rate is %f,weight_scale is %f "
                  "five-layer network to overfit on 50 training examples" % (best_lr, best_wc))
    plt.plot(loss_history)
    plt.title("Training loss history")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()


def batch_normalization():
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    # You should expect losses between 1e-4~1e-10 for W,
    # losses between 1e-08~1e-10 for b,
    # and losses between 1e-08~1e-09 for beta and gammas.
    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                  reg=reg, weight_scale=5e-2, dtype=np.float64,
                                  normalization='batchnorm')

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
        if reg == 0: print()


def Batch_Normalization_for_Deep_Networks():
    np.random.seed(231)

    # Try training a very deep net with batchnorm.
    hidden_dims = [100, 100, 100, 100, 100]

    num_train = 1000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    weight_scale = 2e-2
    bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization='batchnorm')
    model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)

    print('Solver with batch norm:')
    bn_solver = Solver(bn_model, small_data,
                       num_epochs=10, batch_size=50,
                       update_rule='adam',
                       optim_config={
                           'learning_rate': 1e-3,
                       },
                       verbose=True, print_every=20)
    bn_solver.train()

    print('\nSolver without batch norm:')
    solver = Solver(model, small_data,
                    num_epochs=10, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()


def Batch_Normalization_and_Initialization():
    """
    我们现在将进行一个小实验来研究批量归一化和权重初始化的相互作用。

    第一部分将使用不同的权重初始化尺度来训练具有和不具有批量归一化的八层网络。
    第二部分将绘制训练准确性、验证集准确性和训练损失作为权重初始化量表的函数
    """
    np.random.seed(231)

    # Try training a very deep net with batchnorm.
    hidden_dims = [50, 50, 50, 50, 50, 50, 50]
    num_train = 1000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    bn_solvers_ws = {}
    solvers_ws = {}
    weight_scales = np.logspace(-4, 0, num=20)
    for i, weight_scale in enumerate(weight_scales):
        print('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
        bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization='batchnorm')
        model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)

        bn_solver = Solver(bn_model, small_data,
                           num_epochs=10, batch_size=50,
                           update_rule='adam',
                           optim_config={
                               'learning_rate': 1e-3,
                           },
                           verbose=True, print_every=200)
        bn_solver.train()
        bn_solvers_ws[weight_scale] = bn_solver

        solver = Solver(model, small_data,
                        num_epochs=10, batch_size=50,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': 1e-3,
                        },
                        verbose=True, print_every=200)
        solver.train()
        solvers_ws[weight_scale] = solver


def run_batchsize_experiments(normalization_mode):
    np.random.seed(231)

    # Try training a very deep net with batchnorm.
    hidden_dims = [100, 100, 100, 100, 100]
    num_train = 1000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    n_epochs = 10
    weight_scale = 2e-2
    batch_sizes = [5, 10, 50, 500]
    lr = 10 ** (-3.5)
    solver_bsize = batch_sizes[0]

    print('No normalization: batch size = ', solver_bsize)
    model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)
    solver = Solver(model, small_data,
                    num_epochs=n_epochs, batch_size=solver_bsize,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': lr,
                    },
                    verbose=True, print_every=500)
    solver.train()

    bn_solvers = []
    for i in range(len(batch_sizes)):
        b_size = batch_sizes[i]
        print('Normalization: batch size = ', b_size)
        bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=normalization_mode)
        bn_solver = Solver(bn_model, small_data,
                           num_epochs=n_epochs, batch_size=b_size,
                           update_rule='adam',
                           optim_config={
                               'learning_rate': lr,
                           },
                           verbose=True, print_every=500)
        bn_solver.train()
        bn_solvers.append(bn_solver)

    return bn_solvers, solver, batch_sizes


def plot_training_history(title, label, baseline, bn_solvers, plot_fn, bl_marker='.', bn_marker='.', labels=None):
    """utility function for plotting training history"""
    plt.title(title)
    plt.xlabel(label)
    bn_plots = [plot_fn(bn_solver) for bn_solver in bn_solvers]
    bl_plot = plot_fn(baseline)
    num_bn = len(bn_plots)
    for i in range(num_bn):
        label = 'with_norm'
        if labels is not None:
            label += str(labels[i])
        plt.plot(bn_plots[i], bn_marker, label=label)
    label = 'baseline'
    if labels is not None:
        label += str(labels[0])
    plt.plot(bl_plot, bl_marker, label=label)
    plt.legend(loc='lower center', ncol=num_bn + 1)


def batchnorm_layernorm():
    batch_sizes = [5, 10, 50, 500]
    # bn_solvers_bsize, solver_bsize, batch_sizes = run_batchsize_experiments('batchnorm')
    bn_solvers_bsize, solver_bsize, batch_sizes = run_batchsize_experiments('layernorm')

    plt.subplot(2, 1, 1)
    plot_training_history('Training accuracy (Batch Normalization)', 'Epoch', solver_bsize, bn_solvers_bsize,
                          lambda x: x.train_acc_history, bl_marker='-^', bn_marker='-o', labels=batch_sizes)
    plt.subplot(2, 1, 2)
    plot_training_history('Validation accuracy (Batch Normalization)', 'Epoch', solver_bsize, bn_solvers_bsize,
                          lambda x: x.val_acc_history, bl_marker='-^', bn_marker='-o', labels=batch_sizes)

    plt.gcf().set_size_inches(15, 10)
    plt.show()


def three_layer_convolutional_network():
    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

    solver = Solver(
        model,
        data,
        num_epochs=1,
        batch_size=50,
        update_rule='adam',
        optim_config={'learning_rate': 1e-3, },
        verbose=True,
        print_every=20
    )
    solver.train()


if __name__ == '__main__':
    three_layer_convolutional_network()
