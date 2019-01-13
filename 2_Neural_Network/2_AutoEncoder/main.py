import os
import numpy as np
import pandas as pd

import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils import *

# HYPER-PARAMETERS
RUN_NAME = 'run01'
SAVE_DIR = 'models'
NUM_HIDDEN_UNITS = [32, 3, 32]
EPOCHS = 100

# Create directory to save model
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
X, y = load_data(file_name)

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
model.add(Dense(NUM_HIDDEN_UNITS[0], activation='tanh', name='FC_1', input_shape=(num_features,)))
model.add(Dense(NUM_HIDDEN_UNITS[1], activation='tanh', name='FC_2'))
model.add(Dense(NUM_HIDDEN_UNITS[2], activation='tanh', name='FC_3'))
model.add(Dense(num_features, name='output'))
model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['mae'])

# CALLBACKS

# TensorBoard
# Save class labels for color data visualization in TensorBoard
index = pd.read_excel(file_name, index_col=0).index
write_metadata(os.path.join(model_dir, 'metadata.tsv'), index, y)

# Create tensorboard callback
tensorboard = TensorBoard(log_dir=model_dir,
                          embeddings_freq=1,
                          embeddings_layer_names=['FC_2'],
                          embeddings_metadata='metadata.tsv',
                          embeddings_data=X)

# SESSION
model.summary()
model.fit(X_train, X_train, epochs=EPOCHS, batch_size=32, validation_split=0.2,
          callbacks=[tensorboard])

model.save_weights(os.path.join(os.path.join(model_dir), 'wieghts.h5'))
