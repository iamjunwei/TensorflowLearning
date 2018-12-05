import numpy as np
import torch.optim as optim


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


class TwoLayerNet(object):
    def __init__(self
                 ,input_dim=3*32*32
                 ,hidden_dim=100
                 ,num_classes=10
                 ,weight_scale=1e-3):
        self.params = {}
        self.params["W2"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b2"] = np.zeros((hidden_dim, ))
        self.params["W1"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b1"] = np.zeros((num_classes, ))

    def loss(self, X, y, reg):
        loss, grads = 0, {}
        h1_out, h1_cache = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, out_cache = affine_relu_forward(h1_out, self.params["W2"], self.params["b2"])
        loss, dout = softmax_loss(scores, y)
        dout, dw2, db2 = affine_relu_backward(dout, out_cache)
        _, dw1, db1 = affine_relu_backward(dout, h1_cache)
        loss += 0.5 * reg * (np.sum(self.params["W2"] ** 2) + np.sum(self.params["W1"] ** 2))
        dw2 += reg * self.params["W2"]
        dw1 += reg * self.params["W1"]
        grads["W2"] = dw2
        grads["b2"] = db2
        grads["W1"] = dw1
        grads["b1"] = db1


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None

    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == "test":
        out = (x - running_mean) / np.sqrt(running_var + eps) * gamma + beta
    else:
        raise ValueError("Invalid forward batchnorm mode: %s" % mode)
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def batchnorm_backward(dout, cache):
    x, mean, var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout * 1.0, axis=0)
    dx_hat = dout * gamma
    dx_hat_numerator = dx_hat / np.sqrt(var + eps)
    dx_hat_denominator = np.sum(dx_hat * (x - mean), axis=0)
    dx_1 = dx_hat_numerator
    dvar = -0.5 * ((var + eps) ** (-1.5)) * dx_hat_denominator
    dmean = -1.0 * np.sum(dx_hat_numerator, axis=0) + dvar * np.mean(-2.0 * (x - mean), axis=0)
    dx_var = dvar * 2.0 / N * (x - mean)
    dx_mean = dmean * 1.0 / N
    dx = dx_1 + dx_var + dx_mean
    return dx, dgamma, dbeta


def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    da_bn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])
    mask = None
    out = None
    if mode == "train":
        keep_prob = 1 - p
        mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        out = mask * x
    elif mode == "test":
        out = x
    else:
        raise ValueError("Invalid mode: %s" % mode)
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    dx = None
    if mode == "train":
        dx = mask * dout
    elif mode == "test":
        dx = dout
    else:
        raise ValueError("Invalid mode: %s" % mode)
    return dx


class FullyConnectedNet(object):
    def __init__(self
                 ,hidden_dims
                 ,input_dim=3*32*32
                 ,num_classes=10
                 ,dropout=0
                 ,use_batchnorm=False
                 ,reg=0.0
                 ,weight_scale=1e-2
                 ,dtype=np.float64
                 ,seed=None):
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.params["W%d" % (i+1, )] = weight_scale * np.random.randn(input_dim, h_dim)
            self.params["b%d" % (i+1, )] = np.zeros((h_dim, ))
            if use_batchnorm:
                self.params["gamma%d" % (i+1, )] = np.ones((h_dim, ))
                self.params["beta%d" % (i+1, )] = np.zeros((h_dim, ))
            in_dim = h_dim
        self.params["W%d" % (self.num_layers, )] = weight_scale * np.random.randn(in_dim, num_classes)
        self.params["b%d" % (self.num_layers, )] = np.zeros((num_classes, ))
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
        if seed is not None:
            self.dropout_param["seed"] = seed
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = None
        if y is None:
            mode = "test"
        else:
            mode = "train"
        if self.dropout_param is not None:
            self.dropout_param["mode"] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        fc_mix_cache = {}
        if self.use_dropout:
            dp_cache = {}
        out = X
        for i in range(self.num_layers - 1):
            w, b = self.params["W%d" % (i+1, )], self.params["b%d" % (i+1, )]
            if self.use_batchnorm:
                gamma = self.params["gamma%d" % (i+1, )]
                beta = self.params["beta%d" % (i+1, )]
                out, fc_mix_cache[i] = affine_bn_relu_forward(out, w, b, gamma, beta, bn_param)
            else:
                out, fc_mix_cache[i] = affine_relu_forward(out, w, b)
            if self.use_dropout:
                out, dp_cache[i] = dropout_forward(out, self.dropout_param)
        w = self.params["W%d" % (self.num_layers, )]
        b = self.params["b%d" % (self.num_layers, )]
        out, out_cache = affine_forward(out, w, b)
        scores = out
        if mode == "test":
            return scores
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params["W%d" % (self.num_layers, )] ** 2)
        dout, dw, db = affine_backward(dout, out_cache)
        grads["W%d" % (self.num_layers, )] = dw + self.reg * self.params["W%d" % (self.num_layers, )]
        grads["b%d" % (self.num_layers, )] = db + self.reg * self.params["b%d" % (self.num_layers, )]
        for i in range(self.num_layers - 1):
            ri = self.num_layers - 2 - i
            loss += 0.5 * self.reg * np.sum(self.params["W%d" % (ri + 1, )] ** 2)
            if self.use_dropout:
                dout = dropout_backward(dout, dp_cache[ri])
            if self.use_batchnorm:
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, fc_mix_cache[ri])
                grads["gamma%d" % (ri+1, )] = dgamma
                grads["beta%d" % (ri+1, )] = dbeta
            else:
                dout, dw, db = affine_relu_backward(dout, fc_mix_cache[ri])
            grads["W%d" % (ri+1, )] = dw + self.reg * self.params["W%d" % (ri+1, )]
            grads["b%d" % (ri+1, )] = db
        return loss, grads


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))
    next_w = None
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    config["velocity"] = v
    return next_w, config
