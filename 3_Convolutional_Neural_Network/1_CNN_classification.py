import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from sklearn.preprocessing import OneHotEncoder
from utils import load_data

NUM_CLASS = 2

X_train, X_test, y_train, y_test = load_data()

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

