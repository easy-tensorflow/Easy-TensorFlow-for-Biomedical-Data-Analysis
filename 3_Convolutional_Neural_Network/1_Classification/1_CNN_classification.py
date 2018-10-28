import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.metrics import confusion_matrix
from utils import load_data, plot_confusion_matrix


# Hyper-parameters
EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_CLASS = 2

X_train, X_test, y_train, y_test = load_data()
input_shape = X_train.shape[1:]

# Build the model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), strides=1, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(NUM_CLASS, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# The patience parameter is the amount of epochs to check for improvement
early_stop = EarlyStopping(monitor='val_loss', patience=20)

# Start Training
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.2, verbose=1, callbacks=[early_stop])


# Predict
y_pred = model.predict(X_test)
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=['non-nodule', 'nodule'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=['non-nodule', 'nodule'], normalize=True,
                      title='Normalized confusion matrix')
