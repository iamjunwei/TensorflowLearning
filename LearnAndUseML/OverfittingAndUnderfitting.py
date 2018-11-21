# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

# tips to avoid overfit
# 获取更多训练数据。
# 降低网络容量。
# 添加权重正则化。
# 添加丢弃层。

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


# one-hot
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss="binary_crossentropy", metrics=["accuracy", "binary_crossentropy"])
baseline_history = baseline_model.fit(train_data, train_labels, epochs=20,
                   batch_size=512, validation_data=(test_data, test_labels), verbose=2)

smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
smaller_model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss="binary_crossentropy", metrics=["accuracy", "binary_crossentropy"])
smaller_history = smaller_model.fit(train_data, test_labels, epochs=20,
                  batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# bigger_model = keras.Sequential([
#     keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# bigger_model.compile(optimizer=tf.train.AdamOptimizer(),
#                      loss="binary_crossentropy", metrics=["accuracy", "binary_crossentropy"])
# bigger_history = bigger_model.fit(train_data, test_labels, epochs=20,
#                  batch_size=512, validation_data=(test_data, test_labels), verbose=2)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()


# plot_history([('baseline', baseline_history),
#               ('smaller', smaller_history),
#               ('bigger', bigger_history)])

# L1 正则化，其中所添加的代价与权重系数的绝对值（即所谓的权重“L1 范数”）成正比
# L2 正则化，其中所添加的代价与权重系数值的平方（即所谓的权重“L2 范数”）成正比。L2 正则化在神经网络领域也称为权重衰减。不要因为名称不同而感到困惑：从数学角度来讲，权重衰减与 L2 正则化完全相同
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)
plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('l2', l2_model_history)])

# dropout layers
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)
plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])
