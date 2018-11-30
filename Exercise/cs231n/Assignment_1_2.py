from __future__ import print_function
import random
import numpy as np
from Exercise.cs231n.cs231n_assignment.cs231n.data_utils import load_CIFAR10
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
from Exercise.cs231n.cs231n_assignment.cs231n.gradient_check import grad_check_sparse
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers import LinearSVM
import matplotlib.pyplot as plt
import time
import math

cifar10_dir = "./cs231n_assignment/cs231n/datasets/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print("Training data shape: ", X_train.shape)
print("Training labels shape: ", y_train.shape)
print("Test data shape: ", X_test.shape)
print("Test labels shape: ", y_test.shape)

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)
sample_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, sample_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(sample_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype("uint8"))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
plt.show()

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
print("Train data shape: ", X_train.shape)
print("Validation data shape: ", X_val.shape)
print("Test data shape: ", X_test.shape)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print("Train data shape: ", X_train.shape)
print("Validation data shape: ", X_val.shape)
print("Test data shape: ", X_test.shape)

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
print("W shape: ", W.shape)
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
print("loss: %f" % (loss,))

# gradient check
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print("Naive loss: %e computed in %fs" % (loss_naive, toc - tic))
tic = time.time()
loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print("Vectorized loss: %e computed in %fs" % (loss_vectorized, toc - tic))
print("Difference loss: %f" % (loss_naive - loss_vectorized,))
print("Difference grad: %f" % (np.linalg.norm(grad_naive - grad_vectorized, ord="fro"),))

svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=True)
toc = time.time()
print("Training took %fs" % (toc - tic,))
plt.plot(loss_hist)
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()

y_train_pred = svm.predict(X_train)
print("training accuracy %f" % (np.mean(y_train_pred == y_train),))
y_val_pred = svm.predict(X_val)
print("validation accuracy %f" % (np.mean(y_val_pred == y_val),))

# optimize learning_rate and regularization
learning_rates = [2e-7, 0.75e-7, 1.5e-7, 1.25e-7, 0.75e-7]
regularization_strengths = [3e4, 3.24e4, 3.5e4, 3.75e4, 4e4, 4.25e4, 4.5e4, 4.75e4, 5.0e4]
result = {}
best_val = -1
best_svm = None
for rate in learning_rates:
    for strength in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate=rate, reg=strength, num_iters=1500, verbose=True)
        y_train_pred = svm.predict(X_train)
        accuracy_train = np.mean(y_train_pred == y_train)
        y_val_pred = svm.predict(X_val)
        accuracy_val = np.mean(y_val_pred == y_val)
        result[(rate, strength)] = (accuracy_train, accuracy_val)
        if best_val < accuracy_val:
            best_val = accuracy_val
            best_svm = svm
for lr, reg in sorted(result):
    accuracy_train, accuracy_val = result[(lr, reg)]
    print("lr %e, reg %e, training acc %f, val acc %f" % (lr, reg, accuracy_train, accuracy_val))
print("best acc %f" % (best_val, ))

y_test_pred = best_svm.predict(X_test)
accuracy_test = np.mean(y_test_pred == y_test)
print("Test acc: %f" % (accuracy_test, ))
