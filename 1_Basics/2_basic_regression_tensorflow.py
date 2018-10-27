import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
EPOCHS = 500
NUM_HIDDEN_UNITS = 64

# Load the Boston Housing Prices dataset
boston_housing = K.datasets.boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
num_features = X_train.shape[1]

# Shuffle the training set
order = np.argsort(np.random.random(y_train.shape))
X_train = X_train[order]
y_train = y_train[order]

print("Training set: {}".format(X_train.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(y_train.shape))  # 102 examples, 13 features

# Normalize features
# Test data is *not* used when calculating the mean and std
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
train_data = (X_train - mean) / std
test_data = (X_train - mean) / std

# Build the model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, num_features], name='X')
y = tf.placeholder(tf.float32, shape=[None, ], name='Y')


def DenseLayer(inputs, )


# Hidden Layer
W1 = tf.get_variable('W1',
                     dtype=tf.float32,
                     shape=[num_features, NUM_HIDDEN_UNITS],
                     initializer=tf.truncated_normal_initializer(stddev=0.01))
b1 = tf.get_variable('b1',
                     dtype=tf.float32,
                     initializer=tf.constant(0., shape=[1], dtype=tf.float32))
fc1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# Output Layer
W2 = tf.get_variable('W2',
                     dtype=tf.float32,
                     shape=[num_features, 1],
                     initializer=tf.truncated_normal_initializer(stddev=0.01))
b2 = tf.get_variable('b2',
                     dtype=tf.float32,
                     initializer=tf.constant(0., shape=[1], dtype=tf.float32))
fc2 = tf.matmul(x, W2) + b2








print()
