import tensorflow as tf
import numpy as np

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
