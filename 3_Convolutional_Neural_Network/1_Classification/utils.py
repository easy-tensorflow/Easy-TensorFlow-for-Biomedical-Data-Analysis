import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools


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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

