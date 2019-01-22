import tensorflow as tf
import pandas as pd
import numpy as np

train_dataFrame = pd.read_csv("./train.csv")
test_dataFrame = pd.read_csv("./test.csv")

train_dataFrame.drop(columns=["Id"], inplace=True)

numerical_train_dataFrame = train_dataFrame.select_dtypes(exclude=["object"])
category_train_dataFrame = train_dataFrame.select_dtypes(include=["object"])
numerical_train_dataFrame.fillna(0, inplace=True)
category_train_dataFrame.fillna("NONE", inplace=True)

max_series = numerical_train_dataFrame.max()
min_series = numerical_train_dataFrame.min()
numerical_train_dataFrame = (numerical_train_dataFrame - min_series) / (max_series - min_series)

labels = numerical_train_dataFrame["SalePrice"]
numerical_train_dataFrame.drop(columns=["SalePrice"], inplace=True)

numerical_column_names = numerical_train_dataFrame.columns.values
category_column_names = category_train_dataFrame.columns.values

feature_columns = []
for col in numerical_column_names:
    feature_columns.append(tf.feature_column.numeric_column(col))
for col in category_column_names:
    hash_bucket_column = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=100)
    embedding_column = tf.feature_column.embedding_column(hash_bucket_column, dimension=4)
    feature_columns.append(embedding_column)

numerical_train_dataFrame = numerical_train_dataFrame.astype(np.float64)

regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                      hidden_units=[1000],
                                      activation_fn=tf.nn.relu,
                                      loss_reduction=tf.losses.Reduction.MEAN)

print(numerical_train_dataFrame.describe())


def model_input_fn():
    feature_dict = {}
    for k in numerical_column_names:
        feature_dict[k] = numerical_train_dataFrame[k].values
    for k in category_column_names:
        feature_dict[k] = category_train_dataFrame[k].values
    return feature_dict, labels.values


regressor.train(input_fn=lambda: model_input_fn(), steps=2000)
evaluate_result = regressor.evaluate(input_fn=lambda: model_input_fn(), steps=1)
print(evaluate_result["loss"])
