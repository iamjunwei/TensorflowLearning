# https://www.tensorflow.org/guide/eager

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

# tensors and numpy array: auto convert


def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(max_num.numpy()):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        elif int(num % 3) == 0:
            print('Fizz')
        elif int(num % 5) == 0:
            print('Buzz')
        else:
            print(num)
        counter += 1
    return counter


print(fizzbuzz(15))

NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise


def prediction(input, weight, bias):
    return input * weight + bias


def loss(input, weight, bias, output):
    predict = prediction(input, weight, bias)
    return tf.reduce_mean(tf.square(predict - output))


def grad(input, weight, bias, output):
    with tf.GradientTape() as t:
        losses = loss(input, weight, bias, output)
    return t.gradient(losses, [weight, bias])


train_steps = 201
learning_rate = 0.01
W = tfe.Variable(5.)
b = tfe.Variable(10.)
for i in range(train_steps):
    losses = loss(training_inputs, W, b, training_outputs)
    dW, db = grad(training_inputs, W, b, training_outputs)
    W.assign_sub(dW * learning_rate)
    b.assign_sub(db * learning_rate)
    if i % 50 == 0:
        print("step: {}, losses: {}, W: {}, b: {}".format(i, losses, W.numpy(), b.numpy()))

"""
pattern1

for (i, (x, y)) in enumerate(dataset_train):
  # Calculate derivatives of the input function with respect to its parameters.
  grads = grad(model, x, y)
  # Apply the gradient to the model
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

"""

"""
pattern2

for (i, (x, y)) in enumerate(dataset_train):
    # minimize() is equivalent to the grad() and apply_gradients() calls.
    optimizer.minimize(lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())

"""

# specify run on gpu
# with tf.device("/gpu:0"): ...

