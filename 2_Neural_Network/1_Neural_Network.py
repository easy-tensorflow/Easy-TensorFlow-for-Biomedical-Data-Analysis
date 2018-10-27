import numpy as np
import pandas as pd
import urllib.request
import tensorflow as tf
from tensorflow import keras as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

num_hidden_units = [64, 64]

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

num_samples, num_features = X.shape
num_classes = np.max(y)

# Preprocessing the data
# one-hot encode
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(np.expand_dims(y, axis=1))

# Neural Network
model = K.Sequential()
for num_hidden in num_hidden_units:
    model.add(K.layers.Dense(num_hidden, activation='relu'))
model.add(K.layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2,
          verbose=1)

