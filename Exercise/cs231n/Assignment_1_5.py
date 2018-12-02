from Exercise.cs231n.cs231n_assignment.cs231n.features import *
from Exercise.cs231n.cs231n_assignment.cs231n.data_utils import load_CIFAR10
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers.linear_classifier import LinearSVM
from Exercise.cs231n.cs231n_assignment.cs231n.classifiers.neural_net import TwoLayerNet

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

num_color_bins = 10
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)
print(X_train_feats.shape, X_val_feats.shape, X_test_feats.shape)

mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

results = {}
best_val = -1
best_svm = None
learning_rates = [5e-9, 7.5e-9, 1e-8]
regularization_strengths = [(5 + i) * 1e6 for i in range(-3, 4)]
for rs in regularization_strengths:
    for lr in learning_rates:
        svm = LinearSVM()
        svm.train(X_train_feats, y_train, lr, rs, num_iters=2000)
        y_train_pred = svm.predict(X_train_feats)
        acc_train = np.mean(y_train_pred == y_train)
        y_val_pred = svm.predict(X_val_feats)
        acc_val = np.mean(y_val_pred == y_val)
        results[(lr, rs)] = (acc_train, acc_val)
        if best_val < acc_val:
            best_val = acc_val
            best_svm = svm
print("Best validation acc of svm: ", best_val)
y_test_pred = best_svm.predict(X_test_feats)
acc_test = np.mean(y_test_pred == y_test)
print("Test acc of svm: ", acc_test)

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10
best_net = None
best_val = -1
learning_rates = [1e-2, 1e-1, 5e-1, 1, 5]
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]
for rs in regularization_strengths:
    for lr in learning_rates:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
                          num_iters=1500, batch_size=200, learning_rate=lr, reg=rs, verbose=False)
        acc_val = np.mean(net.predict(X_val_feats) == y_val)
        if best_val < acc_val:
            best_val = acc_val
            best_net = net
print("Best validation acc of net: ", best_val)
acc_test = np.mean(best_net.predict(X_test_feats) == y_test)
print("Test acc of net: ", acc_test)
