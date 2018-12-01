from __future__ import print_function
import random
import numpy as np
from Exercise.cs231n.cs231n_assignment.cs231n.data_utils import load_CIFAR10
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from Exercise.cs231n.cs231n_assignment.cs231n.gradient_check import grad_check_sparse
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers import Softmax
import matplotlib.pyplot as plt
import time

cifar10_dir = "./cs231n_assignment/cs231n/datasets/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# pre-processing
mean_image = np.mean(X_train, axis=0)  # mean value of each pixel
X_train = X_train - mean_image
X_val = X_val - mean_image
X_dev = X_dev - mean_image
X_test = X_test - mean_image

# add bias column
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
print(X_train.shape, X_val.shape, X_dev.shape, X_test.shape)

W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
print("softmax naive loss %f" % (loss, ))

f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.0)
toc = time.time()
print("softmax naive computed in %fs" % (toc - tic, ))
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
toc = time.time()
print("softmax vectorized computed in %fs" % (toc - tic, ))
print("Loss difference %f" % (loss_naive - loss_vectorized))
print("Loss grad %f" % (np.linalg.norm(grad_naive - grad_vectorized, ord="fro")))

result = {}
best_val = -1
best_softmax = None
learning_rates = np.logspace(-10, 10, 10)
regularization_strengths = np.logspace(-3, 6, 10)
iters = 100
for lr in learning_rates:
    for rs in regularization_strengths:
        softmax = Softmax()
        softmax.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters, verbose=True)
        y_train_pred = softmax.predict(X_train)
        acc_train = np.mean(y_train_pred == y_train)
        y_val_pred = softmax.predict(X_val)
        acc_val = np.mean(y_val_pred == y_val)
        result[(lr, rs)] = (acc_train, acc_val)
        if best_val < acc_val:
            best_softmax = softmax
            best_val = acc_val
for lr in learning_rates:
    for rs in regularization_strengths:
        acc_train, acc_val = result[(lr, rs)]
        print("lr %e, reg %e, acc train %f, acc val %f" % (lr, rs, acc_train, acc_val))
print("best validation accuracy %f" % (best_val, ))
y_test_pred = best_softmax.predict(X_test)
test_acc = np.mean(y_test_pred == y_test)
print("Test accuracy %f" % (test_acc, ))
