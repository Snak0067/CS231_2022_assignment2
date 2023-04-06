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

        if type(hidden_dims) != list:
            raise ValueError('hidden_dim has to be a list')

        self.L = len(hidden_dims) + 1
        self.N = input_dim
        self.C = num_classes
        dims = [self.N] + hidden_dims + [self.C]
        Ws = {'W' + str(i + 1): weight_scale * np.random.randn(dims[i], dims[i + 1]) for i in range(len(dims) - 1)}
        b = {'b' + str(i + 1): np.zeros(dims[i + 1]) for i in range(len(dims) - 1)}

        self.params.update(b)
        self.params.update(Ws)

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
            self.bn_params = {'bn_param' + str(i + 1): {'mode': 'train',
                                                        'running_mean': np.zeros(dims[i + 1]),
                                                        'running_var': np.zeros(dims[i + 1])}
                              for i in range(len(dims) - 2)}
            gammas = {'gamma' + str(i + 1): np.ones(dims[i + 1]) for i in range(len(dims) - 2)}
            betas = {'beta' + str(i + 1): np.zeros(dims[i + 1]) for i in range(len(dims) - 2)}
            self.params.update(betas)
            self.params.update(gammas)

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
                bn_param[mode] = mode
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

        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        if self.use_dropout:
            # dropout on the input layer
            hdrop, cache_hdrop = dropout_forward(hidden['h0'], self.dropout_param)
            hidden['hdrop0'], hidden['cache_hdrop0'] = hdrop, cache_hdrop

        for i in range(self.L):
            idx = i + 1
            # Naming of the variable
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = hidden['h' + str(idx - 1)]
            if self.use_dropout:
                h = hidden['hdrop' + str(idx - 1)]
            if self.normalization == "batchnorm" and idx != self.L:
                gamma = self.params['gamma' + str(idx)]
                beta = self.params['beta' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]

            # Computing of the forward pass.
            # Special case of the last layer (output)
            if idx == self.L:
                h, cache_h = affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h

            # For all other layers
            else:
                if self.normalization == "batchnorm":
                    h, cache_h = affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                else:
                    h, cache_h = affine_relu_forward(h, w, b)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h

                if self.use_dropout:
                    h = hidden['h' + str(idx)]
                    hdrop, cache_hdrop = dropout_forward(h, self.dropout_param)
                    hidden['hdrop' + str(idx)] = hdrop
                    hidden['cache_hdrop' + str(idx)] = cache_hdrop

        scores = hidden['h' + str(self.L)]

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

        # Computing of the loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss

        # Backward pass

        hidden['dh' + str(self.L)] = dscores
        for i in range(self.L)[::-1]:
            idx = i + 1
            dh = hidden['dh' + str(idx)]
            h_cache = hidden['cache_h' + str(idx)]
            if idx == self.L:
                dh, dw, db = affine_backward(dh, h_cache)
                hidden['dh' + str(idx - 1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

            else:
                if self.use_dropout:
                    # First backprop in the dropout layer
                    cache_hdrop = hidden['cache_hdrop' + str(idx)]
                    dh = dropout_backward(dh, cache_hdrop)
                if self.normalization == "batchnorm":
                    dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx - 1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db
                    hidden['dgamma' + str(idx)] = dgamma
                    hidden['dbeta' + str(idx)] = dbeta
                else:
                    dh, dw, db = affine_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx - 1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db

        # w gradients where we add the regulariation term
        list_dw = {key[1:]: val + self.reg * self.params[key[1:]] for key, val in hidden if key[:2] == 'dW'}
        # Paramerters b
        list_db = {key[1:]: val for key, val in hidden if key[:2] == 'db'}
        # Parameters gamma
        list_dgamma = {key[1:]: val for key, val in hidden if key[:6] == 'dgamma'}
        # Paramters beta
        list_dbeta = {key[1:]: val for key, val in hidden if key[:5] == 'dbeta'}

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
