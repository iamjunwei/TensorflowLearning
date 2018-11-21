# https://www.tensorflow.org/tutorials/eager/custom_training

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

x = tf.zeros([10, 10])
x += 2  # This is equivalent to x = x + 2, which does not mutate the original value of x
print(x)

v = tfe.Variable(1.0)
assert v.numpy() == 1.0

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0


class Model(object):
    def __init__(self):
        self.W = tfe.Variable(5.0)
        self.b = tfe.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()
print(model(3.0).numpy())


def loss(predict_y, desired_y):
    return tf.reduce_mean(tf.square(predict_y - desired_y))


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise
print(loss(model(inputs), outputs).numpy())


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(dW * learning_rate)
    model.b.assign_sub(db * learning_rate)


epochs = range(10)
for epoch in epochs:
    train(model, inputs, outputs, 0.1)
    print(loss(model(inputs), outputs).numpy())
