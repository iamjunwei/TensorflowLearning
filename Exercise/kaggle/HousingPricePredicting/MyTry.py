import tensorflow as tf
import pandas as pd
import numpy as np

train_dataFrame = pd.read_csv("./train.csv")
test_dataFrame = pd.read_csv("./test.csv")
# print(train_dataFrame.head())

train_dataFrame.drop(columns=["Id"], inplace=True)
# print(train_dataFrame.head())

numerical_train_dataFrame = train_dataFrame.select_dtypes(exclude=["object"])
# category_train_dataFrame = train_dataFrame.select_dtypes(include=["object"])
numerical_train_dataFrame.fillna(0, inplace=True)
# print(numerical_train_dataFrame.info())
# category_train_dataFrame.fillna("NONE", inplace=True)
# print(category_train_dataFrame.info())

mean_series = numerical_train_dataFrame.mean()
std_series = numerical_train_dataFrame.std()
numerical_train_dataFrame = (numerical_train_dataFrame - mean_series) / std_series
# print(numerical_train_dataFrame.describe())

labels = numerical_train_dataFrame["SalePrice"]
numerical_train_dataFrame.drop(columns=["SalePrice"], inplace=True)

numerical_column_names = numerical_train_dataFrame.columns.values
# print(numerical_column_names)
# category_column_names = category_train_dataFrame.columns.values

# print(numerical_train_dataFrame.info())
# print(category_train_dataFrame.info())
# print(len(labels.values))

feature_columns = []
for col in numerical_column_names:
    feature_columns.append(tf.feature_column.numeric_column(col))
# for col in category_column_names:
#     hash_bucket_column = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=100)
#     embedding_column = tf.feature_column.embedding_column(hash_bucket_column, dimension=4)
#     feature_columns.append(embedding_column)

# print(numerical_train_dataFrame.info())
# print(numerical_train_dataFrame["1stFlrSF"])
# print("--------")
# print(len(numerical_train_dataFrame.columns.values))
numerical_train_dataFrame = numerical_train_dataFrame.astype(np.float64)

regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                      hidden_units=[1000],
                                      activation_fn=tf.nn.relu)

print(numerical_train_dataFrame.describe())

def model_input_fn():
    feature_dict = {}
    for col in numerical_column_names:
        feature_dict[col] = numerical_train_dataFrame[col].values
    # for col in category_column_names:
    #     feature_dict[col] = category_train_dataFrame[col].values
    return feature_dict, labels.values


regressor.train(input_fn=lambda: model_input_fn(), steps=2000)
evaluate_result = regressor.evaluate(input_fn=lambda: model_input_fn(), steps=1)
print(evaluate_result["loss"])
