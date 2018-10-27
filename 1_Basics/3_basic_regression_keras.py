import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
EPOCHS = 500
NUM_HIDDEN_UNITS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Load the Boston Housing Prices dataset
boston_housing = K.datasets.boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
num_features = X_train.shape[1]

# Shuffle the training set
order = np.argsort(np.random.random(y_train.shape))
X_train = X_train[order]
y_train = y_train[order]

print("Training set: {}".format(X_train.shape))   # 404 examples, 13 features
print("Testing set:  {}".format(y_train.shape))   # 102 examples, 13 features

# Normalize features
# Test data is *not* used when calculating the mean and std
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Build the model
model = K.Sequential()
model.add(K.layers.Dense(NUM_HIDDEN_UNITS, activation='relu', input_shape=(num_features,)))
model.add(K.layers.Dense(1, activation='linear'))
model.compile(loss='mse',
              optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
              metrics=['mae'])
model.summary()

# The patience parameter is the amount of epochs to check for improvement
early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.2, verbose=1, callbacks=[early_stop])


plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [1000$]')
plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
plt.legend()
plt.ylim([0, 5])


[loss, mae] = model.evaluate(X_test, y_test, verbose=0)
print("\n Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# Predict
y_pred = model.predict(X_test).flatten()

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100], color='red')

plt.figure()
error = y_pred - y_test
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [1000$]")
plt.ylabel("Count")
plt.show()
print()
