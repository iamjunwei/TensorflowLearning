from __future__ import print_function
import random
import numpy as np
from Exercise.cs231n.cs231n_assignment.cs231n.data_utils import load_CIFAR10
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
import matplotlib.pyplot as plt

from past.builtins import xrange


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

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)
plt.imshow(dists, interpolation="none")
plt.show()

y_test_pred = classifier.predict_labels(dists, k=1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = num_correct / num_test
print("Got %d / %d correct => accuracy: %f" % (num_correct, num_test, accuracy))

dists_one = classifier.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord="fro")
print("Difference was: %f" % (difference,))

dists_two = classifier.compute_distances_no_loops(X_test)
difference = np.linalg.norm(dists - dists_two, ord="fro")
print("Difference was: %f" % (difference,))


def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


print("Two loop cost: ", time_function(classifier.compute_distances_two_loops, X_test))
print("One loop cost: ", time_function(classifier.compute_distances_one_loop, X_test))
print("No loop cost: ", time_function(classifier.compute_distances_no_loops, X_test))

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
k_to_accuracies = {}
classifier = KNearestNeighbor()
for k in k_choices:
    accuracies = np.zeros(num_folds)
    for fold in xrange(num_folds):
        temp_X = X_train_folds[:]
        temp_y = y_train_folds[:]
        X_validation_fold = temp_X.pop(fold)
        y_validation_fold = temp_y.pop(fold)
        temp_X = np.array([y for x in temp_X for y in x])
        temp_y = np.array([y for x in temp_y for y in x])
        classifier.train(temp_X, temp_y)
        y_validation_pred = classifier.predict(X_validation_fold, k=k)
        num_correct = np.sum(y_validation_pred == y_validation_fold)
        accuracies[fold] = float(num_correct) / len(y_validation_fold)
    k_to_accuracies[k] = accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print("k = %d, accuracy = %f" % (k, accuracy))

best_k = 10
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct ) / len(y_test)
print("Got %d / %d correct => accuracy: %f" % (num_correct, len(y_test), accuracy))
