# https://www.tensorflow.org/tutorials/eager/automatic_differentiation
# 自动微分

import tensorflow as tf
from math import pi
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tf.enable_eager_execution()


# Derivatives of a function
def f(x):
    return tf.square(tf.sin(x))


def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]


assert f(pi / 2).numpy() == 1.0
grad_f = tfe.gradients_function(f)
assert tf.abs(grad_f(pi / 2)[0]).numpy() < 1e-7
print("continue.....")

x = tf.lin_space(-2*pi, 2*pi, 100)
plt.plot(x, f(x), label="f")
plt.plot(x, grad(f)(x), label="first derivative")
plt.plot(x, grad(grad(f))(x), label="second derivative")
plt.plot(x, grad(grad(grad(f)))(x), label="third derivative")
plt.legend()
plt.show()

# GradientTape
x = tf.ones((2, 2))
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dy = t.gradient(z, y)  # z对y求导
dz_dx = t.gradient(z, x)  # z对x求导
print(dz_dy.numpy())
print(dz_dx.numpy())

# Higher-order gradients
x = tfe.Variable(1.0)
with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)    # 3x2
d2y_dx2 = t.gradient(dy_dx, x)   # 6x
print(dy_dx.numpy())
print(d2y_dx2.numpy())
