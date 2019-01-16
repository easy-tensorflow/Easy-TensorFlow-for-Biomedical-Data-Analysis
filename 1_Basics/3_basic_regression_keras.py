from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from utils import randomize

# Hyper-parameters
EPOCHS = 500
NUM_HIDDEN_UNITS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Load the Boston Housing Prices dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
num_features = X_train.shape[1]

# Shuffle the training set
X_train, y_train = randomize(X_train, y_train)

print("Train data size -> input: {}, output: {}".format(X_train.shape, y_train.shape))
print("Test data size: -> input: {}, output: {}".format(X_test.shape, y_test.shape))

# Normalize features
# Test data is *not* used when calculating the mean and std
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Build the model
model = Sequential()
model.add(Dense(NUM_HIDDEN_UNITS, activation='relu', input_shape=(num_features,)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE), metrics=['mae'])

# Let's print a summary of the network structure
model.summary()

# Stop training when a monitored quantity has stopped improving
# The patience parameter is the amount of epochs to check for improvement
early_stop = EarlyStopping(monitor='val_loss', patience=20)

# Start Training
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.2, verbose=1, callbacks=[early_stop])

# Let's plot the error values through training epochs
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [1000$]')
plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
plt.legend()
plt.ylim([0, 5])

# Let's check the error value on test data
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
