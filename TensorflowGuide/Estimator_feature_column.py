# https://www.tensorflow.org/guide/feature_columns
import tensorflow as tf

# numerical column
# Defaults to a tf.float32 scalar.
numeric_feature_column1 = tf.feature_column.numeric_column(key="SepalLength")
numeric_feature_column2 = tf.feature_column.numeric_column(key="SepalLength",
                                                           dtype=tf.float64)


# bucket column
# bucket0: below 1960
# bucket1: 1960-1980
# bucket2: 1980-2000
# bucket3: above 2000
numeric_feature_column3 = tf.feature_column.numeric_column("Year")
bucketized_feature_column = tf.feature_column.bucketized_column(source_column=numeric_feature_column3,
                                                                boundaries=[1960, 1980, 2000])


# category_column_with_identity
# assume value from 0-4
# 0: [1, 0, 0, 0]
# 1: [0, 1, 0, 0]
# 2: [0, 0, 1, 0]
# 3: [0, 0, 0, 1]
# dataset must contains feature named with "my_feature_b" which value from 0 to 4
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4)  # Values [0, 4)


# category_column_with_vocabulary_list
vocabulary_feature_column_list = tf.feature_column.categorical_column_with_vocabulary_list(
    key="my_feature_b",
    vocabulary_list=["kitchenware", "electronics", "sports"])


# category_column_with_vocabulary_file
# each line in vocabulary file is a word
vocabulary_feature_column_file = tf.feature_column.categorical_column_with_vocabulary_file(
        key="my_feature_b",
        vocabulary_file="product_class.txt",
        vocabulary_size=3)


# category_column_with_hash_bucket
# when category number is too large, we can use hash number to bucketize feature
hashed_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
        key="some_feature",
        hash_buckets_size=100)  # The number of categories


# cross column
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(...))
longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    list(...))
# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)


# indicator column
# one-hot
categorical_column = ...  # any type of category column
indicator_column = tf.feature_column.indicator_column(categorical_column)


# embedding column
# 多种类的值映射到一个向量，训练过程中，学习这个映射关系
number_of_categories = 81
embedding_dimensions = number_of_categories ** 0.25
embedding_column = tf.feature_column.embedding_column(categorical_column, embedding_dimensions)


# LinearClassifier: all type of feature columns
# LinearRegressor: all type of feature columns
# DNNClassifier: only dense column, using indicator column and embedding column
# DNNRegressor: only dense column, using indicator column and embedding column
# DNNLinearCombinedClassifier: linear_feature_columns for all type, dnn_feature_columns for dense column
# DNNLinearCombinedRegressor: linear_feature_columns for all type, dnn_feature_columns for dense column
