import os
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt

# Read Data
train = pd.read_csv("./train.csv")
train = train.select_dtypes(exclude=["object"])
print(train.head())
train.drop(columns=["Id"], inplace=True)
print(train.head())
train.fillna(0, inplace=True)

test = pd.read_csv("./test.csv")
ID = test.Id
test = test.select_dtypes(exclude=["object"])
test.drop(columns=["Id"], inplace=True)
test.fillna(0, inplace=True)

# Outliers
clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns=["Top"])
train = train.iloc[y_noano[y_noano["Top"] == 1].index.values]
train.reset_index(drop=True, inplace=True)

# Pre-processing
col_train = list(train.columns)
col_train_bis = list(train.columns)
col_train_bis.remove("SalePrice")
mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop('SalePrice',axis = 1))
mat_y = np.array(train.SalePrice).reshape((1314,1))
prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)
prepro = MinMaxScaler()
prepro.fit(mat_train)
prepro_test = MinMaxScaler()
prepro_test.fit(mat_test)
train = pd.DataFrame(prepro.transform(mat_train), columns=col_train)
test = pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_bis)
print(train.shape)
print(train.head())

COLUMNS = col_train
FEATURES = col_train_bis
LABEL = "SalePrice"
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
training_set = train[COLUMNS]
prediction_set = train[LABEL]
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES], prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns=[LABEL])
training_set = pd.DataFrame(x_train, columns=FEATURES).merge(y_train, left_index=True, right_index=True)
print(training_set.shape)
print(training_set.head())
y_test = pd.DataFrame(y_test, columns=[LABEL])
testing_set = pd.DataFrame(x_test, columns=FEATURES).merge(y_test, left_index=True, right_index=True)
print(testing_set.shape)
print(testing_set.head())

# Deep Neural Network for continuous features
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          activation_fn=tf.nn.relu,
                                          hidden_units=[200, 100, 50, 25, 12])
training_set.reset_index(drop=True, inplace=True)


def input_fn(data_set, pred=False):
    if pred == False:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return feature_cols, labels
    else:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        return feature_cols


regressor.fit(input_fn=lambda : input_fn(training_set), steps=2000)
ev = regressor.evaluate(input_fn=lambda : input_fn(testing_set), steps=1)
loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))
y = regressor.predict(input_fn=lambda :input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
print(list(np.round(predictions, 8)))
print(list(np.round(y_test["SalePrice"], 8)))

# prediction and submission
predictions = pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(y_test.shape[0], 1)), columns=["Prediction"])
reality = pd.DataFrame(prepro.inverse_transform(testing_set), columns=[COLUMNS]).SalePrice
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

fig, ax = plt.subplots(figsize=(10, 8))

plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel('Predictions', fontsize = 12)
plt.ylabel('Reality', fontsize = 12)
plt.title('Predictions x Reality on dataset Test', fontsize = 12)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

y_predict = regressor.predict(input_fn=lambda :input_fn(test, pred=True))


def to_submit(pred_y, name_out):
    y_predict = list(itertools.islice(pred_y, test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(test.shape[0], 1)), columns=["Prediction"])
    y_predict = y_predict.join(ID)
    y_predict.to_csv(name_out + ".csv", index=False)


to_submit(y_predict, "submission_continuous")
