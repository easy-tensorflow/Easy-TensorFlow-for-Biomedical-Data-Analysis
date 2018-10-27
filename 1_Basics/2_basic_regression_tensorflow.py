import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split

# Hyper-parameters
EPOCHS = 500
NUM_HIDDEN_UNITS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DISP_FREQ = 100

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
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

# Build the model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, num_features], name='X')
y = tf.placeholder(tf.float32, shape=[None], name='Y')


def DenseLayer(inputs, num_units, layer_name, activation=None):
    input_dim = inputs.shape[1]
    with tf.variable_scope(layer_name):
        W = tf.get_variable('W',
                            dtype=tf.float32,
                            shape=[input_dim, num_units],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b',
                            dtype=tf.float32,
                            initializer=tf.constant(0., shape=[num_units], dtype=tf.float32))
        logits = tf.matmul(inputs, W) + b
        if activation:
            return activation(logits)
    return logits


# Hidden Layer
fc1 = DenseLayer(x, NUM_HIDDEN_UNITS, 'FC1', activation=tf.nn.relu)

# Output Layer
predictions = DenseLayer(fc1, 1, 'FC2')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=tf.squeeze(predictions)))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# Create an interactive session (to keep the session in the other cells)
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)
    # Number of training iterations in each epoch
    NUM_TR_ITERS = int(len(y_train) / BATCH_SIZE)
    print('------------------------------------')
    for epoch in range(EPOCHS):
        # Randomly shuffle the training data at the beginning of each epoch
        x_train, y_train = randomize(X_train, y_train)
        for iteration in range(NUM_TR_ITERS):
            start = iteration * BATCH_SIZE
            end = (iteration + 1) * BATCH_SIZE
            x_batch, y_batch = get_next_batch(X_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)
        if not epoch % DISP_FREQ:
            # Run validation after every epoch
            feed_dict_valid = {x: X_valid, y: y_valid}
            loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
            print("Epoch: {0}, validation loss: {1:.2f}".format(epoch + 1, loss_valid))
            print('------------------------------------')
