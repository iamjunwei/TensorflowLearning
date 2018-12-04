import numpy as np


def affine_forward(x, w, b):
    reshape_x = np.reshape(x, (x.shape[0], -1))
    out = reshape_x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def softmax_loss(z, y):
    probs = np.exp(z - np.max(z, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = z.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dz = probs.copy()
    dz[np.arange(N), y] -= 1
    dz /= N
    return loss, dz


def affine_backward(dout, cache):
    z, w, b = cache
    reshaped_x = np.reshape(z, (z.shape[0], -1))
    dz = np.reshape(dout.dot(w.T), z.shape)
    dw = np.dot(reshaped_x.T, dout)
    db = np.sum(dout, axis=0)
    return dz, dw, db


def relu_backward(dout, cache):
    dx, x = None, cache
    dx = (x > 0) * dout
    return dx


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


