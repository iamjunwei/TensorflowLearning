# https://www.tensorflow.org/tutorials/keras/save_and_restore_models
# 用于创建模型的代码
# 模型的训练权重或参数

import tensorflow as tf
from tensorflow import keras

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_data = train_data[:1000].reshape(-1, 28*28) / 255.0
test_data = test_data[:1000].reshape(-1, 28*28) / 255.0


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28*28,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"],
                  optimizer=keras.optimizers.Adam())
    return model


# use callback to automatically save checkpoints
model = create_model()
checkpoint_path = "./cp.ckpt"
checkpoint_dir = "."
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels),
          callbacks=[cp_callback])

model = create_model()
loss, acc = model.evaluate(test_data, test_labels)
print("acc of untrained model: ", acc)

model = create_model()
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_data, test_labels)
print("acc of trained model: ", acc)

# 检查点包括：
# 包含模型权重的一个或多个分片
# 指示哪些权重存储在哪些分片中的索引文件

# manually save checkpoints
model = create_model()
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
model.save_weights("./manual_cp")
model = create_model()
model.load_weights("./manual_cp")
loss, acc = model.evaluate(test_data, test_labels)
print("acc of manual cp model: ", acc)

# save the whole model, including weights, net structure and optimizers
model.save("./model.h5")
model = keras.models.load_model("./model.h5")
loss, acc = model.evaluate(test_data, test_labels)
print("acc of loaded model: ", acc)
