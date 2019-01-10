# https://www.tensorflow.org/guide/estimators
import tensorflow as tf


# 预创建的Estimator的程序结构
# 1、编写一个或多个数据集导入函数：包含一个字典（键是特征名称，值是特征tensor）和一个标签tensor
def input_fn(dataset):
    # manipulate dataset, extracting the feature dict and the label
    pass  # return feature_dict, label


# 2、创建特征列：tf.feature_column
global_education_mean = 0
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                                                    normalizer_fn=lambda x: x - global_education_mean)

# 3、实例化预创建的Estimator
estimator = tf.estimator.LinearClassifier(feature_columns=[population, crime_rate, median_education])

# 4、训练、评估、推理
estimator.train(input_fn=input_fn, steps=2000)

# 实际Estimator的工作流程：先在多个预创建的Estimator中选择最佳的，再尝试使用自定义Estimator

# keras to estimator
train_data = None
train_labels = None
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                           loss='categorical_crossentropy',
                           metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)
# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
input_names = keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
