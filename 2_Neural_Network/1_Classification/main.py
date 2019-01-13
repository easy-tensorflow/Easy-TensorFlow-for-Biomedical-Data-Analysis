import os
import numpy as np
import pandas as pd

import keras
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from utils import *

MODE = 'test'
RUN_NAME = 'run01'
SAVE_DIR = 'models'
NUM_HIDDEN_UNITS = [70, 30]
EPOCHS = 500

model_dir = os.path.join(SAVE_DIR, RUN_NAME)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# LOAD DATA
# Download the Mice Protein Expression dataset from uci
# https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
file_name = 'Data_Cortex_Nuclear.xls'
download_data(url, file_name)

# Load the dataset
X, y = load_data(file_name, rm_nan_by_axis=1)


num_samples, num_features = X.shape
num_classes = np.max(y) + 1


# PREPROCESSING
# One-hot encode
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(np.expand_dims(y, axis=1))

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=.3)


# BUILD MODEL
model = Sequential()
model.add(Dense(NUM_HIDDEN_UNITS[0], activation='relu', name='FC_1', input_shape=(num_features,)))
model.add(Dense(NUM_HIDDEN_UNITS[1], activation='relu', name='FC_2'))
model.add(Dense(num_classes, activation='softmax', name='output'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# CALLBACKS
# Early Stop
early_stop = EarlyStopping(monitor='val_loss', patience=20)

# TensorBoard
tensorboard = TensorBoard(log_dir=model_dir)

# SESSION
model.summary()
if MODE == 'train':
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2,
              callbacks=[tensorboard, early_stop])

    model.save_weights(os.path.join(os.path.join(model_dir), 'wieghts.h5'))

    loss, acc = model.evaluate(X_test, y_test)
    print('Testing set Loss: {:.2f}'.format(loss))
    print('Testing set Accuracy: {:.2%}'.format(acc))

if MODE == 'test':
    # Predict
    model.load_weights(os.path.join(os.path.join(model_dir), 'wieghts.h5'))
    y_pred = model.predict(X_test)
    classes = list(pd.read_excel(file_name, index_col=0)['class'].astype('category').cat.categories)
    cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plot_confusion_matrix(cnf_matrix, classes=classes)
