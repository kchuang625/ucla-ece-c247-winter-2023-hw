import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
        use_batchnorm=False,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize the weights and biases of a three layer CNN. To initialize:
        #     - the biases should be initialized to zeros.
        #     - the weights should be initialized to a matrix with entries
        #         drawn from a Gaussian distribution with zero mean and
        #         standard deviation given by weight_scale.
        # ================================================================ #

        C, H, W = input_dim

        # CNN layer
        stride = 1
        pad = (filter_size - 1) / 2
        self.params["W1"] = np.random.normal(
            0, weight_scale, [num_filters, C, filter_size, filter_size]
        )
        self.params["b1"] = np.zeros([num_filters])

        # FC1
        h_out_cnn = (H - filter_size + 2 * pad) / stride + 1
        w_out_cnn = (W - filter_size + 2 * pad) / stride + 1
        h_out_pooling = int((h_out_cnn - 2) / 2 + 1)
        w_out_pooling = int((w_out_cnn - 2) / 2 + 1)
        self.params["W2"] = np.random.normal(
            0, weight_scale, [h_out_pooling * w_out_pooling * num_filters, hidden_dim]
        )
        self.params["b2"] = np.zeros([hidden_dim])

        # FC2
        self.params["W3"] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params["b3"] = np.zeros([num_classes])

        # batch norm layers
        if self.use_batchnorm:
            self.bn_params = []
            # CNN
            self.params["gamma1"] = np.ones(num_filters)
            self.params["beta1"] = np.zeros(num_filters)
            self.bn_params.append({"mode": "train"})
            # FC1
            self.params["gamma2"] = np.ones(hidden_dim)
            self.params["beta2"] = np.zeros(hidden_dim)
            self.bn_params.append({"mode": "train"})

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the three layer CNN.  Store the output
        #   scores as the variable "scores".
        # ================================================================ #

        if self.use_batchnorm:
            # set mode
            mode = "test" if y is None else "train"
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

            # get parameters
            gamma1, gamma2 = self.params["gamma1"], self.params["gamma2"]
            beta1, beta2 = self.params["beta1"], self.params["beta2"]
            bn_param1, bn_param2 = self.bn_params

            # foward CNN and FC1 layers
            out, cnn_cache = conv_bn_relu_pool_forward(
                X, W1, b1, conv_param, gamma1, beta1, bn_param1, pool_param
            )
            out, fc1_cache = affine_bn_relu_forward(out, W2, b2, gamma2, beta2, bn_param2)
        else:
            out, cnn_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            out, fc1_cache = affine_relu_forward(out, W2, b2)

        scores, fc2_cache = affine_forward(out, W3, b3)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        if y is None:
            return scores

        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the three layer CNN.  Store the grads
        #   in the grads dictionary, exactly as before (i.e., the gradient of
        #   self.params[k] will be grads[k]).  Store the loss as "loss", and
        #   don't forget to add regularization on ALL weight matrices.
        # ================================================================ #

        # compute loss
        loss, dout = softmax_loss(scores, y)

        # add regularization loss
        for i in range(3):
            W = self.params["W" + str(i + 1)]
            loss += 0.5 * self.reg * (W * W).sum()

        # compute gradients
        dout, dw3, db3 = affine_backward(dout, fc2_cache)
        grads["W3"], grads["b3"] = dw3 + self.reg * W3, db3

        if self.use_batchnorm:
            dout, dw2, db2, dgamma2, dbeta2 = affine_bn_relu_backward(dout, fc1_cache)
            _, dw1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(dout, cnn_cache)
            grads["gamma1"], grads["gamma2"] = dgamma1, dgamma2
            grads["beta1"], grads["beta2"] = dbeta1, dbeta2
        else:
            dout, dw2, db2 = affine_relu_backward(dout, fc1_cache)
            _, dw1, db1 = conv_relu_pool_backward(dout, cnn_cache)

        grads["W2"], grads["b2"] = dw2 + self.reg * W2, db2
        grads["W1"], grads["b1"] = dw1 + self.reg * W1, db1

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads
