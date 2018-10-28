import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from utils import load_data

# Hyper-parameters
EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_CLASS = 2

X_train, X_test, y_train, y_test = load_data()
input_shape = X_train.shape[1:]

# Build the model
model = K.Sequential()
model.add(K.layers.Conv2D(16, kernel_size=(3, 3), strides=1, activation='relu', input_shape=input_shape))
model.add(K.layers.MaxPool2D(pool_size=(2, 2)))
model.add(K.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(K.layers.MaxPool2D(pool_size=(2, 2)))
model.add(K.layers.Flatten())
model.add(K.layers.Dense(100, activation='relu'))
model.add(K.layers.Dense(NUM_CLASS, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# The patience parameter is the amount of epochs to check for improvement
early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Start Training
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.2, verbose=1, callbacks=[early_stop])
