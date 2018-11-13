# High-level Api: Estimators
# Mid-level Api: Layers, Datasets, Metrics
# Low-level Api: Python
# Tensorflow Distributed Execution Engine

import tensorflow as tf
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']


def load_data(label_name="Species"):
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split("/")[-1], origin=TRAIN_URL)
    train = pd.read_csv(filepath_or_buffer=train_path, names=CSV_COLUMN_NAMES, header=0)
    train_features, train_label = train, train.pop(label_name)
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split("/")[-1], origin=TEST_URL)
    test = pd.read_csv(filepath_or_buffer=test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)
    return (train_features, train_label), (test_features, test_label)


(train_features, train_label), (test_features, test_label) = load_data()

my_feature_columns = [
    tf.feature_column.numeric_column(key="SepalLength"),
    tf.feature_column.numeric_column(key="SepalWidth"),
    tf.feature_column.numeric_column(key="PetalLength"),
    tf.feature_column.numeric_column(key="PetalWidth"),
]


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size=batch_size)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    if labels is None:
        inputs = dict(features)
    else:
        inputs = (dict(features), labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[10, 10], n_classes=3)
classifier.train(input_fn=lambda: train_input_fn(train_features, train_label, 32), steps=1000)
eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_features, test_label, 32))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
