import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import matplotlib.pyplot as plt

# Load the Boston Housing Prices dataset
boston_housing = K.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

# Normalize features
# Test data is *not* used when calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# Create the model
def build_model():
    model = K.Sequential([K.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
                          K.layers.Dense(64, activation=tf.nn.relu),
                          K.layers.Dense(1)])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model


model = build_model()
model.summary()


# Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 500
# The patience parameter is the amount of epochs to check for improvement
early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


def plot_history(hstry):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(hstry.epoch, np.array(hstry.history['mean_absolute_error']), label='Train Loss')
    plt.plot(hstry.epoch, np.array(hstry.history['val_mean_absolute_error']), label='Val loss')
    plt.legend()
    plt.ylim([0, 5])


plot_history(history)
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("\n Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# Predict
test_predictions = model.predict(test_data).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [1000$]")
plt.ylabel("Count")

plt.show()
print()
