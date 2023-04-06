from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """
        Initialize a new FullyConnectedNet

        Args:
            hidden_dims:一个整数列表，给出每个隐藏层大小
            input_dim:一个整数，给出输入维度大小
            num_classes:一个整数，给出要分类的类的数量
            dropout_keep_ratio: 介于 0_1 之间的标量，给出 dropout强度,如果 dropout_keep_ratio=1，则网络不使用dropout
            normalization:网络使用正则化的类型. 有效值为: "batchnorm", "layernorm", or None 表示无正则化(默认值)
            reg:一个标量，给出L2正则化强度。
            weight_scale:标量，给出权重随机初始化的标准偏差
            dtype:numpy数据类型对象；所有计算都将使用此数据类型执行,float32速度更快，但精度较低，因此应该使用 float64进行数值梯度检查.
            seed:If not None, 则将此随机种子传递到 Dropout层,这将使 Dropout层具有决定性，因此我们可以对模型进行梯度检查
        """

        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for i in range(len(hidden_dims)):
            D = hidden_dims[i]
            self.params['W' + str(i + 1)] = weight_scale * np.random.randn(input_dim, D)
            self.params['b' + str(i + 1)] = np.zeros(D)
            if normalization is not None and normalization == 'batchnorm':
                self.params['gamma' + str(i + 1)] = np.ones(D)
                self.params['beta' + str(i + 1)] = np.zeros(D)
            input_dim = D
        self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(input_dim, num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache = {}
        cache_dropout = {}
        input_x = X
        for i in range(self.num_layers - 1):
            W = self.params['W' + str(i + 1)]
            b = self.params['b' + str(i + 1)]
            # 如果要用batch normalization，一般是在ReLU层前用
            # if self.normalization == "batchnorm":
            #     gamma = self.params['gamma' + str(i + 1)]
            #     beta = self.params['beta' + str(i + 1)]
            #     bn_param = self.bn_params[i]
            #     input_x, cache[i] = conv_bn_relu_forward(input_x, W, b, gamma, beta, bn_param)
            input_x, cache[i] = affine_relu_forward(input_x, W, b)
            if self.use_dropout:
                input_x, cache_dropout[i] = dropout_forward(input_x, self.dropout_param)
        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        scores, cache[self.num_layers - 1] = affine_forward(input_x, W, b)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscore = softmax_loss(scores, y)
        W = self.params['W' + str(self.num_layers)]
        loss += self.reg * 0.5 * np.sum(W * W)
        dhidden, dw, db = affine_backward(dscore, cache[self.num_layers - 1])
        grads['W' + str(self.num_layers)] = dw + self.reg * W
        grads['b' + str(self.num_layers)] = db

        for i in range(self.num_layers - 1, 0, -1):
            W = self.params['W' + str(i)]
            loss += self.reg * 0.5 * np.sum(W ** 2)
            # dropout层在ReLU层后面，所以先计算它的反向求导
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, cache_dropout[i - 1])
            else:
                dhidden, dw, db = affine_relu_backward(dhidden, cache[i - 1])
            grads['W' + str(i)] = dw + self.reg * W
            grads['b' + str(i)] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
