from __future__ import print_function
import numpy as np
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers.neural_net import TwoLayerNet
from Exercise.cs231n.cs231n_assignment.cs231n.gradient_check import eval_numerical_gradient
from Exercise.cs231n.cs231n_assignment.cs231n.data_utils import load_CIFAR10

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


net = init_toy_model()
X, y = init_toy_data()

scores = net.loss(X)
print("Scores : ", scores)
correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.85042150],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949],
])
print("Difference score: ", np.sum(np.abs(scores - correct_scores)))

loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133
print("Difference loss: ", np.abs(loss - correct_loss))


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


loss, grads = net.loss(X, y, reg=0.05)
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

net = init_toy_model()
stats = net.train(X, y, X, y, learning_rate=1e-1, reg=5e-6, num_iters=100, verbose=False)
print("Final training loss: ", stats["loss_history"][-1])


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
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# pre-processing
mean_image = np.mean(X_train, axis=0)  # mean value of each pixel
X_train = X_train - mean_image
X_val = X_val - mean_image
X_test = X_test - mean_image

print(X_train.shape, X_val.shape, X_test.shape)

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)
stats = net.train(X_train, y_train, X_val, y_val, num_iters=1000, batch_size=200,
                  learning_rate=1e-4, learning_rate_decay=0.95, reg=0.25, verbose=True)
val_acc = np.mean(net.predict(X_val) == y_val)
print("Validation acc: ", val_acc)

best_net = None
best_val = -1
best_stats = None
learning_rates = [1e-2, 1e-3]
regularization_strengths = [0.4, 0.5, 0.6]
results = {}
iters = 2000
for lr in learning_rates:
    for rs in regularization_strengths:
        net = TwoLayerNet(input_size, hidden_size, num_classes)
        stats = net.train(X_train, y_train, X_val, y_val, learning_rate=lr, reg=rs, num_iters=iters, verbose=False)
        y_train_pred = net.predict(X_train)
        acc_train = np.mean(y_train_pred == y_train)
        y_val_pred = net.predict(X_val)
        acc_val = np.mean(y_val_pred == y_val)
        results[(lr, rs)] = (acc_train, acc_val)
        if best_val < acc_val:
            best_net = net
            best_stats = stats
            best_val = acc_val
print("Best val acc: ", best_val)
acc_test = np.mean(best_net.predict(X_test) == y_test)
print("Test acc: ", acc_test)
