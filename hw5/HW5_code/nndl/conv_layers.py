import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of a convolutional neural network.
    #   Store the output as 'out'.
    #   Hint: to pad the array, you can use the function np.pad.
    # ================================================================ #

    N, _C, H, W = x.shape
    F, _C, HH, WW = w.shape

    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_height = int(1 + (H + 2 * pad - HH) / stride)
    out_width = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros([N, F, out_height, out_width])

    for i in range(N):
        for c_i in range(F):
            for h_i in range(out_height):
                for w_i in range(out_width):
                    out[i, c_i, h_i, w_i] = (
                        w[c_i]
                        * xpad[
                            i,
                            :,
                            h_i * stride : h_i * stride + HH,
                            w_i * stride : w_i * stride + WW,
                        ]
                    ).sum() + b[c_i]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache

    stride, pad = [conv_param["stride"], conv_param["pad"]]
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    num_filts, _, f_height, f_width = w.shape

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #

    # dx with padding
    dx = np.zeros_like(xpad)

    for i in range(N):
        for c_i in range(F):
            for h_i in range(out_height):
                for w_i in range(out_width):
                    dx[
                        i,
                        :,
                        h_i * stride : h_i * stride + f_height,
                        w_i * stride : w_i * stride + f_width,
                    ] += (
                        dout[i, c_i, h_i, w_i] * w[c_i]
                    )

    # adjust dx shape
    H, W = x.shape[-2:]
    dx = dx[:, :, pad : H + pad, pad : W + pad]

    # dw
    dw = np.zeros_like(w)

    for i in range(N):
        for c_i in range(F):
            for h_i in range(out_height):
                for w_i in range(out_width):
                    dw[c_i] += (
                        dout[i, c_i, h_i, w_i]
                        * xpad[
                            i,
                            :,
                            h_i * stride : h_i * stride + f_height,
                            w_i * stride : w_i * stride + f_width,
                        ]
                    )

    # db
    db = dout.sum(axis=(0, 2, 3))

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape
    out_height = int((H - pool_height) / stride + 1)
    out_width = int((W - pool_width) / stride + 1)
    out = np.zeros([N, C, out_height, out_width])

    for i in range(N):
        for c_i in range(C):
            for h_i in range(out_height):
                for w_i in range(out_width):
                    out[i, c_i, h_i, w_i] = (
                        x[
                            i,
                            c_i,
                            h_i * stride : h_i * stride + pool_height,
                            w_i * stride : w_i * stride + pool_width,
                        ]
                    ).max()

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = (
        pool_param["pool_height"],
        pool_param["pool_width"],
        pool_param["stride"],
    )

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #

    N, C = x.shape[:2]
    out_height, out_width = dout.shape[-2:]
    dx = np.zeros_like(x)

    for i in range(N):
        for c_i in range(C):
            for h_i in range(out_height):
                for w_i in range(out_width):
                    max_idx_1d = np.argmax(
                        x[
                            i,
                            c_i,
                            h_i * stride : h_i * stride + pool_height,
                            w_i * stride : w_i * stride + pool_width,
                        ]
                    )
                    max_idx_2d = np.unravel_index(max_idx_1d, [pool_height, pool_width])
                    dx[
                        i,
                        c_i,
                        h_i * stride + max_idx_2d[0],
                        w_i * stride + max_idx_2d[1],
                    ] = dout[i, c_i, h_i, w_i]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you
    #   implemented in HW #4.
    # ================================================================ #

    N, C, H, W = x.shape
    x_flatten = x.transpose(0, 2, 3, 1).reshape((N * H * W, C))
    out, cache = batchnorm_forward(x_flatten, gamma, beta, bn_param)
    out = out.reshape((N, H, W, C)).transpose(0, 3, 1, 2)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you
    #   implemented in HW #4.
    # ================================================================ #

    N, C, H, W = dout.shape
    dout_flatten = dout.transpose((0, 2, 3, 1)).reshape((N * H * W, C))
    dx, dgamma, dbeta = batchnorm_backward(dout_flatten, cache)
    dx = dx.reshape((N, H, W, C)).transpose(0, 3, 1, 2)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dgamma, dbeta
