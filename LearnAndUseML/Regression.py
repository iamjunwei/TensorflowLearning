# https://www.tensorflow.org/tutorials/keras/basic_regression
# 均方误差 (MSE) 是用于回归问题的常见损失函数（与分类问题不同）。
# 同样，用于回归问题的评估指标也与分类问题不同。常见回归指标是平均绝对误差 (MAE)。
# 如果输入数据特征的值具有不同的范围，则应分别缩放每个特征。
# 如果训练数据不多，则选择隐藏层较少的小型网络，以避免出现过拟合。
# 早停法是防止出现过拟合的实用技术。

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print(train_data[0])

train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)

print(train_data[0])

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=tf.train.RMSPropOptimizer(0.001), metrics=["mae"])
model.summary()

history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, verbose=0)
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("mae on test data: ", mae * 1000)

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
