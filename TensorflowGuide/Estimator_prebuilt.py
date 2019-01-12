# https://www.tensorflow.org/guide/premade_estimators
import tensorflow as tf

# 1 创建一个或多个输入函数。
# 2 定义模型的特征列。
# 3 实例化 Estimator，指定特征列和各种超参数。
# 4 在 Estimator 对象上调用一个或多个方法，传递适当的输入函数作为数据的来源。


# 输入函数，返回dataset
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


# 创建特征列tf.feature_column
my_feature_columns = []
# for key in train_x.keys():
#    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# 实例化Estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# 训练、评估、预测
# classifier.train(
#     input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
#     steps=args.train_steps)
# eval_result = classifier.evaluate(
#     input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))
# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
# expected = ['Setosa', 'Versicolor', 'Virginica']
# predict_x = {
#     'SepalLength': [5.1, 5.9, 6.9],
#     'SepalWidth': [3.3, 3.0, 3.1],
#     'PetalLength': [1.7, 4.2, 5.4],
#     'PetalWidth': [0.5, 1.5, 2.1],
# }
# predictions = classifier.predict(
#     input_fn=lambda:iris_data.eval_input_fn(predict_x,
#                                             batch_size=args.batch_size))
# template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
# for pred_dict, expec in zip(predictions, expected):
#     class_id = pred_dict['class_ids'][0]
#     probability = pred_dict['probabilities'][class_id]
#     print(template.format(iris_data.SPECIES[class_id],
#                           100 * probability, expec))
