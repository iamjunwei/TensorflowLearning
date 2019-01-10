# https://www.tensorflow.org/guide/keras

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
# kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。

# Create a sigmoid layer:
keras.layers.Dense(64, activation='sigmoid')
# Or:
keras.layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
keras.layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
keras.layers.Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
keras.layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))

model.compile(tf.train.AdamOptimizer(), loss=keras.losses.categorical_crossentropy, metrics=tf.metrics.accuracy)

# optimizer：此对象会指定训练过程。从 tf.train 模块向其传递优化器实例，例如 AdamOptimizer、RMSPropOptimizer 或 GradientDescentOptimizer。
# loss：要在优化期间最小化的函数。常见选择包括均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。
# 损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。
# metrics：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))
model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset, validation_steps=3)

model.evaluate(val_data, val_labels, batch_size=32, steps=30)
model.predict(val_data, batch_size=32, steps=30)

# callbacks
# tf.keras.callbacks.ModelCheckpoint：定期保存模型的检查点。
# tf.keras.callbacks.LearningRateScheduler：动态更改学习速率。
# tf.keras.callbacks.EarlyStopping：在验证效果不再改进时中断训练。
# tf.keras.callbacks.TensorBoard：使用 TensorBoard 监控模型的行为
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./my_model')
# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('my_model')
# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')
# Restore the model's state
model.load_weights('my_model.h5')


# Save architecture
# Serialize a model to JSON format
json_string = model.to_json()
# Recreate the model (freshly initialized)
fresh_model = keras.models.from_json(json_string)
# Serializes a model to YAML format
yaml_string = model.to_yaml()
# Recreate the model
fresh_model = keras.models.from_yaml(yaml_string)


# Save all
# Create a trivial model
model = keras.Sequential([
  keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)
# Save entire model to a HDF5 file
model.save('my_model.h5')
# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model('my_model.h5')

# Keras 支持 EagerExecution

# Keras model 可转换成Estimator tf.keras.estimator.model_to_estimator(model)

# 分布式训练
# first，create a keras model
# second, create a estimator RunConfig from MirroredStrategy
# third, convert keras model to estimator
# fourth, estimator train with dataset input function
