import numpy as np
import h5py
from sklearn.model_selection import train_test_split


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
    X = h5f['x'][:]
    y = h5f['y'][:]
    h5f.close()

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # Normalize the input data
    m = np.mean(X_train)
    s = np.std(X_train)
    X_train = (X_train - m) / s
    X_test = (X_test - m) / s

    return X_train, X_test, y_train, y_test

