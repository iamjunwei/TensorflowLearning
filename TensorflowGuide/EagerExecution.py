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

w = tfe.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w
grads = tape.gradient(loss, w)
print(grads)

# Training model process as follows:
# first, create dataset
# second, create keras model
# third, foreach single data
#           with tf.gradientType() as tape:
#               logits = model(input),
#               loss = cal(logits, labels)
#           grads = tape.gradient(loss, variables)
#           optimizer.apply_gradient(zip(grads, variables), global_step=...)

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


# Variable lifecycle
# variable will be cleaned when last reference has been removed


# save checkpoints
x = tfe.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)
checkpoint.save("./ckpt/")
x.assign(11.0)  # x is 11.0
checkpoint.restore(tf.train.latest_checkpoint("./ckpt/"))
print(x)  # x is 10.0
# save all
# checkpoint = tf.train.Checkpoint(optimizer=...,
#                                  model=...,
#                                  optimizer_step=tf.train.get_or_create_global_step())


# tfe.metrics, pass values to update metrics
m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # 2.5
m([8, 9])
m.result()  # 5.5


# tfe.gradients_function and tfe.value_and_gradients_function
def square(x):
    return tf.multiply(x, x)


gradss = tfe.gradients_function(square)
gradss(3.)[0].numpy()  # gradss returns a tensor list which are corresponding to param list


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

# eager execution: better debugging and interactive performance
# graph execution: better distribution-training and deploy performance
# also, we can save checkpoints in eager execution and use it again in graph execution

# tfe.py_func: eager execution in graph execution environment
def my_py_func(x):
    x = tf.matmul(x, x)  # You can use tf ops
    print(x)  # but it's eager!
    return x

with tf.Session() as sess:
    x = tf.placeholder(dtype=tf.float32)
    # Call eager function in graph!
    pf = tfe.py_func(my_py_func, [x], tf.float32)
    sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
