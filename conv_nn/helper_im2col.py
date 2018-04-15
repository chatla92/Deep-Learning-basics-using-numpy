import numpy as np
from im2col import *

def relu(Z):
    A = np.maximum(0.0, Z)
    cache = {"Z": Z}
    return A, cache


def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ


def linear(Z):
    A = Z
    cache = {}
    return A, cache


def linear_der(dA, cache):
    dZ = np.array(dA, copy=True)
    return dZ


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = {"A": A}
    return Z, cache


def layer_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

    cache = {"lin_cache": lin_cache, "act_cache": act_cache}
    return A, cache


def linear_backward(dZ, cache, W, b):
    A_prev = cache["A"]
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def softmax_cross_entropy_loss(Z, Y=np.array([])):
    max_s = np.max(Z, axis=0)
    softmax = np.exp(Z - max_s)
    A = softmax / softmax.sum(axis=0)
    cache = {"A": A}

    if Y.size == 0:
        loss = []
    else:
        A = A+1e-15
        loss = -np.log(A[Y.astype(int), np.arange(Y.shape[0])]).mean().sum(axis=-1)
    return A, cache, loss


def softmax_cross_entropy_loss_der(Y, cache):
    dZ = cache["A"]
    dZ[[Y.astype(int), np.arange(Y.shape[0])]] -= 1
    dZ = dZ / float(Y.shape[0])
    return dZ


def conv_layer_forward(A_prev, parameters):
    W = parameters['W1']
    b = parameters['b1']
    stride = parameters['stride']
    pad = parameters['pad']

    (m, Ht_prev, Wd_prev, Ch_prev) = A_prev.shape
    (f, f, Ch_prev, Ch) = W.shape

    Ht = int((Ht_prev - f + (2 * pad)) / stride) + 1
    Wd = int((Wd_prev - f + (2 * pad)) / stride) + 1

    W = W.transpose(3, 2, 0, 1)
    A_prev = A_prev.transpose(0, 3, 1, 2)
    b = b.reshape(Ch, 1)

    X_col = im2col_indices(A_prev, f, f, padding=pad, stride=stride)
    W_col = W.reshape(Ch, -1)
    out = np.dot(W_col, X_col) + b
    out = out.reshape(Ch, Ht, Wd, m)
    out = out.transpose(3, 1, 2, 0)

    cache = (A_prev, W, b, stride, pad, X_col)

    return out, cache


def conv_layer_backward(dZ, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape
    db = np.sum(dZ, axis=(0, 1, 2))

    dout_reshaped = dZ.transpose(0, 3, 1, 2).reshape(n_filter, -1)
    dW = np.dot(dout_reshaped, X_col.T)
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = np.dot(W_reshape.T, dout_reshaped)
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)
    dW = dW.transpose(2, 3, 1, 0)
    db = db.reshape(1, 1, 1, n_filter)
    return dX, dW, db


def pool_forward(A_prev, parameters):
    (m, Ht_prev, Wd_prev, Ch_prev) = A_prev.shape

    f = parameters["pool_size"]
    stride = parameters["pool_stride"]

    Ht = int(1 + (Ht_prev - f) / stride)
    Wd = int(1 + (Wd_prev - f) / stride)
    Ch = Ch_prev

    A = np.zeros((m, Ht, Wd, Ch))

    for i in range(m):
        for h in range(0, Ht, stride):
            for w in range(0, Wd, stride):
                for c in range(Ch):
                    a_prev_slice = A_prev[h:h + f, w:w + f, :]
                    A[i, h, w, c] = np.max(a_prev_slice)
    cache = (A_prev, parameters)
    return A, cache


def pool_backward(dA, cache):
    (A_prev, parameters) = cache
    stride = parameters['pool_stride']
    f = parameters['pool_size']

    (m, Ht_prev, Wd_prev, Ch_prev) = A_prev.shape
    (m, Ht, Wd, Ch) = dA.shape
    dA_prev = np.zeros((m, Ht_prev, Wd_prev, Ch_prev))

    for i in range(m):

        a_prev = A_prev[i]

        for h in range(0, Ht, stride):
            for w in range(0, Wd, stride):
                for c in range(Ch):
                    a_prev_slice = a_prev[h:h + f, w:w + f, c]
                    mask = a_prev_slice == np.max(a_prev_slice)
                    dA_prev[i, h:h + f, w:w + f, c] += mask * dA[i, h, w, c]
    return dA_prev


def dropout_forward(A_prev, prob=0.3):
    P = np.random.rand(A_prev.shape[0], A_prev.shape[1])
    P = P < prob
    A = np.multiply(A_prev, P)
    A = A/prob
    cache = {"P": P, "prob": prob}
    return A, cache


def dropout_backward(dA, cache):
    dA_prev = dA * cache['P']
    dA_prev = dA_prev / cache['prob']
    return dA_prev