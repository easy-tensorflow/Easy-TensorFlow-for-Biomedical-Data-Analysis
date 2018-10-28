import os
import numpy as np
import pandas as pd
import urllib.request

import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

RUN_NAME = 'run02'
SAVE_DIR = 'models'
NUM_HIDDEN_UNITS = [64, 32]
LEARNING_RATE = 0.001
EPOCHS = 500

model_dir = os.path.join(SAVE_DIR, RUN_NAME)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# PREPARE DATASET==============================================================================
# Download the Mice Protein Expression dataset from uci
# https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
file_name = 'Data_Cortex_Nuclear.xls'
urllib.request.urlretrieve(url, file_name)

# Prepare dataset for training
xls_file = pd.read_excel(file_name, index_col=0)
# remove missing values by row: axis=0, column: axis=1
xls_file = xls_file.dropna(axis=1)

X = xls_file[xls_file.columns[0:-4]].values
y = xls_file['class'].astype('category').cat.codes.values

num_samples, num_features = X.shape
num_classes = np.max(y) + 1

# PREPROCESSING=================================================================================
# one-hot encode
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(np.expand_dims(y, axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# NEURAL NETWORK===============================================================================
# Build Model
model = Sequential()
model.add(Dense(NUM_HIDDEN_UNITS[0], activation='relu', name='FC_1', input_shape=(num_features,)))
model.add(Dense(NUM_HIDDEN_UNITS[1], activation='relu', name='FC_2'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])

# TensorBoard
# create metadata for projection
with open(os.path.join(model_dir, 'metadata.tsv'), 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in zip(xls_file.index, np.argmax(y_test, axis=1)):
        f.write("{}\t{}\n".format(index, label))

tensorboard = TensorBoard(log_dir=model_dir,
                          embeddings_freq=1,
                          embeddings_layer_names=['FC_2'],
                          embeddings_metadata='metadata.tsv',
                          embeddings_data=X_test)

checkpoints = ModelCheckpoint(os.path.join(model_dir + 'weightss.hdf5'), monitor='val_loss',
                              verbose=1, save_best_only=True, mode='max')

model.summary()
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2,
          callbacks=[tensorboard, checkpoints])


loss, acc = model.evaluate(X_test, y_test)
print('Testing set Loss: {:.2f}'.format(loss))
print('Testing set Accuracy: {:.2%}'.format(acc))

model.save_weights(os.path.join(os.path.join(SAVE_DIR, RUN_NAME), 'wieghts.h5'))

