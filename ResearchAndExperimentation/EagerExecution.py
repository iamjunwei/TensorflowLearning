# https://www.tensorflow.org/tutorials/eager/eager_basics

import tensorflow as tf
import numpy as np


tf.enable_eager_execution()

# Tensors
# dataType, shape
# able to exist in accelerator memory (GPU)
# immutable
print(tf.add(1, 2))                             # tf.Tensor(3, shape=(), dtype=int32)
print(tf.add([1, 2], [3, 4]))                   # tf.Tensor([4 6], shape=(2,), dtype=int32)
print(tf.square(5))                             # tf.Tensor(25, shape=(), dtype=int32)
print(tf.reduce_sum([1, 2, 3]))                 # tf.Tensor(6, shape=(), dtype=int32)
print(tf.encode_base64("hello world"))          # tf.Tensor(b'aGVsbG8gd29ybGQ', shape=(), dtype=string)
print(tf.square(2) + tf.square(3))              # tf.Tensor(13, shape=(), dtype=int32)

ndarray = np.ones((3, 3))
tensor = tf.multiply(ndarray, 42)  # automatically convert numpy array to tensor
print(tensor)
print(np.add(tensor, 1))  # automatically convert tensor to numpy array
print(tensor.numpy())

# GPU acceleration
x = tf.random_uniform(shape=(3, 3))
print(tf.test.is_gpu_available())
print(x.device.endswith("GPU:0"))  # tensor.device. the 0 index of gpu

# tensorflow will automatically choose one device to run
# but you can choose specific device to run
with tf.device("CPU:0"):
    print("Run on CPU:0")
    x = tf.random_uniform((3, 3))
    tf.matmul(x, x)
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        print("Run on GPU:0")
        x = tf.random_uniform((3, 3))
        tf.matmul(x, x)

# Create a source dataset
# Dataset.from_tensors, Dataset.from_tensor_slices, TextLineDataset(from file), TFRecordDataset(from file)
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])
# ds_file = tf.data.TextLineDataset(filenames=...)

# Apply transformations for dataset
square_ds_tensors = ds_tensors.map(tf.square)
shuffle_ds_tensors = square_ds_tensors.shuffle(2)
batch_ds_tensors = shuffle_ds_tensors.batch(2)

# Iterate for dataset
for x in ds_tensors:
    print(x)
print("----------------")
for x in square_ds_tensors:
    print(x)
print("----------------")
for x in shuffle_ds_tensors:
    print(x)
print("----------------")
for x in batch_ds_tensors:
    print(x)
