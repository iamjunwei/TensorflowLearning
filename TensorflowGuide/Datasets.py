# https://www.tensorflow.org/guide/datasets

import tensorflow as tf
import numpy as np

# tf.data.Dataset： 包含一系列元素。每个元素包含一个或多个Tensor
# tf.data.Iterator： 提供从数据集中提取元素的方法

# tf.data.Dataset.from_tensors / tf.data.Dataset.from_tensor_slices
# for TFRecord file：tf.data.TFRecordDataset

# Dataset.map：每个元素应用一个函数
# Dataset.batch：多个元素批量组成一组元素

# Dataset.make_one_shot_iterator：每次访问一个元素的迭代器
# Iterator.initializer：初始化迭代器
# Iterator.get_next：获取下一个元素

# 返回每个元素的类型和size
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

# 为元素中每个tensor命名
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

# map / flat_map / filter
dataset1 = dataset1.map(lambda x: ...)
dataset2 = dataset2.flat_map(lambda x, y: ...)
# Note: Argument destructuring is not available in Python 3.
# dataset3 = dataset3.filter(lambda x, (y, z): ...)

# 单次迭代器
with tf.Session() as sess1:
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element1 = iterator.get_next()

# 可初始化迭代器
with tf.Session() as sess2:
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element2 = iterator.get_next()
    sess2.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess2.run(next_element2)
        assert i == value

# 可重新初始化迭代器：使用不同的Dataset进行初始化
with tf.Session() as sess3:
    training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element3 = iterator.get_next()
    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess3.run(training_init_op)
        for _ in range(100):
            sess3.run(next_element3)

        # Initialize an iterator over the validation dataset.
        sess3.run(validation_init_op)
        for _ in range(50):
            sess3.run(next_element3)

# 可馈送迭代器：使用handle作为构造方法，传入其它数据集iterator的handle
with tf.Session() as sess4:
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)
    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle,
                                                   training_dataset.output_types,
                                                   training_dataset.output_shapes)
    next_element = iterator.get_next()
    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess4.run(training_iterator.string_handle())
    validation_handle = sess4.run(validation_iterator.string_handle())
    # Run 200 steps using the training dataset. Note that the training dataset is
    # infinite, and we resume from where we left off in the previous `while` loop
    # iteration.
    for _ in range(200):
        sess4.run(next_element, feed_dict={handle: training_handle})
    # Run one pass over the validation dataset.
    sess4.run(validation_iterator.initializer)
    for _ in range(50):
        sess4.run(next_element, feed_dict={handle: validation_handle})

# 获取下一个元素
with tf.Session() as sess5:
    iterator = dataset2.make_one_shot_iterator()
    next1, next2 = iterator.get_next()
    while True:
        try:
            sess5.run(next1, next2)
        except tf.errors.OutOfRangeError:
            break

# 保存和恢复iterator
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()
with tf.Session() as sess6:
    saver.save(sess6, "./ckpt")
    saver.restore(sess6, "./ckpt")

# 读取numpy数据
with tf.Session as sess7:
    features = np.zeros((500, 10))
    labels = np.zeros((500, ))
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    iterator = dataset.make_initializable_iterator()
    sess7.run(iterator.initializer, feed_dict={features_placeholder: features,
                                               labels_placeholder: labels})

# 读取文本数据
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset_text = dataset.flat_map(
    lambda filename:
        tf.data.TextLineDataset(filename).skip(1).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#")))

# 读取csv数据
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
dataset_csv = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2, 4])

# 读取TFRecord数据
with tf.Session() as sess8:
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset_tfrecord = tf.data.TFRecordDataset(filenames)
    iterator = dataset_tfrecord.make_initializable_iterator()
    training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    sess8.run(iterator.initializer, feed_dict={filenames: training_filenames})
    validation_filenames = ["/var/data/validation1.tfrecord", ...]
    sess8.run(iterator.initializer, feed_dict={filenames: validation_filenames})

# dataset.map(some_function)，some_function若包含非tensorflow操作，最好使用tf.py_func包装一下

# dataset.batch：批量组合元素，元素之间shape相同
# dataset.padded_batch：批量组合元素，元素之间shape可以不同，使用特定值填充缺失维度

dataset.repeat(count=10)  # count设置数据集的重复次数，无参数或者None表示无限循环，在数据重新开始时，不会收到通知
dataset.shuffle(buffer_size=10000)  # 维护一个缓冲区，从缓冲区随机选择一个数据

# 高阶API，MonitoredTrainingSession，使用sess.should_stop判断数据集是否结束
