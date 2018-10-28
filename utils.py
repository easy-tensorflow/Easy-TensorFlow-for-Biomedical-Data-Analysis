import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def load_data():
    h5f = h5py.File('LUNA_2d.h5', 'r')
    x = h5f['x'][:][:, :, :, np.newaxis]
    y = h5f['y'][:]
    h5f.close()

    # Shuffle the training set
    x, y = randomize(x, y)

    # one-hot encode the labels
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(np.expand_dims(y, axis=1))

    # split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

    print("Training set: {}".format(x_train.shape))  # 1600 samples
    print("Testing set:  {}".format(x_test.shape))   # 400 samples

    # Normalize the input data
    m = np.mean(x_train)
    s = np.std(x_train)
    x_train = (x_train - m) / s
    x_test = (x_test - m) / s
    return x_train, x_test, y_train, y_test

