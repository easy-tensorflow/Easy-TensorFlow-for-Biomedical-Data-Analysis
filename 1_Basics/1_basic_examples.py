import tensorflow as tf
import numpy as np


# Example 1:
import tensorflow as tf
a = 2
b = 3
c = tf.add(a, b, name='Add')
print(c)


# Example 2:
x = 2
y = 3
add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')


# Example 3:
x = 2
y = 3
add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')
useless_op = tf.multiply(x, add_op, name='Useless')

with tf.Session() as sess:
    pow_out = sess.run(pow_op)


# Example 4: _Constant_
a = tf.constant(2, name='A')
b = tf.constant(3, name='B')
c = tf.add(a, b, name='Add')

with tf.Session() as sess:
    print(sess.run(c))


# Example 5: _Variable_
# create graph
a = tf.get_variable(name="A", initializer=tf.constant([[0, 1], [2, 3]]))
b = tf.get_variable(name="B", initializer=tf.constant([[4, 5], [6, 7]]))
c = tf.add(a, b, name="Add")

# launch the graph in a session
with tf.Session() as sess:
    # now we can run the desired operation
    print(sess.run(c))


# Example 6: _Placeholder_
import tensorflow as tf
a = tf.constant([5, 5, 5], tf.float32, name='A')
b = tf.placeholder(tf.float32, shape=[3], name='B')
c = tf.add(a, b, name="Add")

with tf.Session() as sess:
    # create a dictionary:
    d = {b: [1, 2, 3]}
    # feed it to the placeholder
    print(sess.run(c, feed_dict=d))


# Example 7: _Math Equation_
x = tf.placeholder(tf.int32, shape=[None], name='x')
y = tf.placeholder(tf.int32, shape=[None], name='y')
c = tf.constant(2, name='c')
x_2 = tf.pow(x, 2, name='x_2')
add_op = tf.add(y, c, name='Add')
mul_op = tf.multiply(x_2, y, name='Multiply')
output = tf.add(add_op, mul_op, name='Output')

with tf.Session() as sess:
    out = sess.run(output, feed_dict={x: np.array([2, 3]), y: np.array([3, 3])})
    print('Output is: {}'.format(out))
