import itertools
import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt


def download_data(url, filename):
    """
    Download the dataset from the url
    :param url: url of file to be downloaded
    :param filename: filname to be saved
    :return:
    """
    urllib.request.urlretrieve(url, filename)


def load_data(filename, class_col='class', rm_nan_by_axis=0):
    """
    Load the dataset from file and return X, y
    :param filename: name of xls file
    :param class_col: column name of class
    :param rm_nan_by_axis: remove empty values by axis row=0, column=1
    :return: X: features y:labels
    """
    xls_file = pd.read_excel(filename, index_col=0)
    # remove missing values by row: axis=0, column: axis=1
    xls_file = xls_file.dropna(axis=rm_nan_by_axis)

    X = xls_file[xls_file.columns[0:-4]].values
    y = xls_file[class_col].astype('category').cat.codes.values

    return X, y


def write_metadata(filename,indices, labels):
    """
    Create a metadata file consisting of sample indices and labels
    :param filename: name of the file to save on disk
    :param shape: tensor of labels
    """
    with open(filename, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in zip(indices, labels):
            f.write("{}\t{}\n".format(index, label))


def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    :param cm: confusion matrix
    :param classes: list of class names
    :param normalize: normalize to 0-1
    :param title: plot title
    :param cmap: colormap
    :return:
    """

    if classes is None:
        classes = ['class_{}'.format(i) for i in range(1, len(cm) + 1)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
