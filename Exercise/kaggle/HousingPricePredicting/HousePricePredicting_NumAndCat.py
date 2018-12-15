import tensorflow as tf
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt

# Deep Neural Network for continuous and categorical features
train = pd.read_csv("./train.csv")
train.drop("Id", axis=1, inplace=True)
train_numerical = train.select_dtypes(exclude=["object"])
train_numerical.fillna(0, inplace=True)
train_categoric = train.select_dtypes(include=["object"])
train_categoric.fillna("NONE", inplace=True)
train = train_numerical.merge(train_categoric, left_index=True, right_index=True)
test = pd.read_csv("./test.csv")
ID = test.Id
test.drop("Id", axis=1, inplace=True)
test_numerical = test.select_dtypes(exclude=["object"])
test_numerical.fillna(0, inplace=True)
test_categoric = test.select_dtypes(include=["object"])
test_categoric.fillna("NONE", inplace=True)
test = test_numerical.merge(test_categoric, left_index=True, right_index=True)
clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(train_numerical)
y_noano = clf.predict(train_numerical)
y_noano = pd.DataFrame(y_noano, columns=["Top"])
train_numerical = train_numerical.iloc[y_noano[y_noano["Top"] == 1].index.values]
train_numerical.reset_index(drop=True, inplace=True)
train_categoric = train_categoric.iloc[y_noano[y_noano["Top"] == 1].index.values]
train_categoric.reset_index(drop=True, inplace=True)
train = train.iloc[y_noano[y_noano["Top"] == 1].index.values]
train.reset_index(drop=True, inplace=True)

col_train_num = list(train_numerical.columns)
col_train_num_bis = list(train_numerical.columns)
col_train_cat = list(train_categoric.columns)
col_train_num_bis.remove("SalePrice")
mat_train = np.matrix(train_numerical)
mat_test = np.matrix(test_numerical)
mat_new = np.matrix(train_numerical.drop("SalePrice", axis=1))
mat_y = np.matrix(train.SalePrice)
prepro_y = MinMaxScaler()
prepro_y.fit(mat_y.reshape(1314, 1))
prepro = MinMaxScaler()
prepro.fit(mat_train)
prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)
train_num_scale = pd.DataFrame(prepro.transform(mat_train), columns=col_train_num)
test_num_scale = pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_num_bis)
train[col_train_num] = pd.DataFrame(prepro.transform(mat_train), columns=col_train_num)
test[col_train_num_bis] = pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_num_bis)

# numerical and categorical features -> engineered features
COLUMNS = col_train_num
FEATURES = col_train_num_bis
LABEL = "SalePrice"
FEATURES_CAT = col_train_cat
engineered_features = []
for continuous_feature in FEATURES:
    engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))
for categorical_feature in FEATURES_CAT:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)
    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column,
                                                                  dimension=16,
                                                                  combiner="sum"))

training_set = train[FEATURES + FEATURES_CAT]
prediction_set = train.SalePrice
x_train, x_test, y_train, y_test = train_test_split(training_set, prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns=[LABEL])
training_set = pd.DataFrame(x_train, columns=FEATURES + FEATURES_CAT)\
    .merge(y_train, left_index=True, right_index=True)
print(FEATURES + FEATURES_CAT)
training_sub = training_set[FEATURES + FEATURES_CAT]
testing_sub = test[FEATURES + FEATURES_CAT]
y_test = pd.DataFrame(y_test, columns=[LABEL])
testing_set = pd.DataFrame(x_test, columns=FEATURES + FEATURES_CAT)\
    .merge(y_test, left_index=True, right_index=True)
training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)
testing_set[FEATURES_CAT] = testing_set[FEATURES_CAT].applymap(str)


def input_fn_new(data_set, training=True):
    continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(data_set[k].size)],
                                           values=data_set[k].values,
                                           dense_shape=[data_set[k].size, 1]) for k in FEATURES_CAT}
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
    if training == True:
        label = tf.constant(data_set[LABEL].values)
        return feature_cols, label
    return feature_cols


regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=tf.nn.relu,
                                          hidden_units=[200, 100, 50, 25, 12])
categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(training_set[k].size)],
                                       values=training_set[k].values,
                                       dense_shape=[training_set[k].size, 1]) for k in FEATURES_CAT}
regressor.fit(input_fn=lambda: input_fn_new(training_set), steps=2000)
ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training=True), steps=1)
loss_score = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score))

y = regressor.predict(input_fn=lambda: input_fn_new(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
predictions = pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(434, 1)),
                           columns=["Prediction"])
reality = pd.DataFrame(prepro.inverse_transform(testing_set[COLUMNS]), columns=[COLUMNS]).SalePrice

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

fig, ax = plt.subplots(figsize=(10, 8))

plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Reality', fontsize=12)
plt.title('Predictions x Reality on dataset Test', fontsize=12)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

y_predict = regressor.predict(input_fn=lambda: input_fn_new(testing_sub, training=False))


def to_submit(pred_y, name_out):
    y_predict = list(itertools.islice(pred_y, test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(test.shape[0], 1)),
                             columns=["Prediction"])
    y_predict = y_predict.join(ID)
    y_predict.to_csv(name_out + ".csv", index=False)


to_submit(y_predict, "submission_continuous_and_category")

# Shallow Network
regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=tf.nn.relu,
                                          hidden_units=[1000])
regressor.fit(input_fn=lambda: input_fn_new(training_set), steps=2000)
ev = regressor.evaluate(input_fn=lambda: input_fn_new(training_set, training=True), steps=1)
loss_score_shallow = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score_shallow))
y_predict = regressor.predict(input_fn=lambda: input_fn_new(testing_sub, training=False))
to_submit(y_predict, "submission_continuous_and_category_shallow")
