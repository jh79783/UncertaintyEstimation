# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import pandas as pd
#
# inputs = keras.Input(shape=(784,), name="digits")
# x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
# x = layers.Dense(64, activation="relu", name="dense_2")(x)
# outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
#
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# # dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# # column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
# #                 'Acceleration', 'Model Year', 'Origin']
# # raw_dataset = pd.read_csv(dataset_path, names=column_names,
# #                       na_values = "?", comment='\t',
# #                       sep=" ", skipinitialspace=True)
# #
# # dataset = raw_dataset.copy()
# # Preprocess the data (these are NumPy arrays)
# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255
#
# y_train = y_train.astype("float32")
# y_test = y_test.astype("float32")
#
# # Reserve 10,000 samples for validation
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]
#
# model.compile(
#     optimizer=keras.optimizers.RMSprop(),  # Optimizer
#     # Loss function to minimize
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     # List of metrics to monitor
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
#
# print("Fit model on training data")
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=2,
#     # We pass some validation for
#     # monitoring validation loss and metrics
#     # at the end of each epoch
#     validation_data=(x_val, y_val),
# )
#
# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
#
# # Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test)
# print("predictions shape:", predictions.shape)
# print("prediction vale:", predictions)

import matplotlib.pyplot as plt
import numpy as np
import math
def normal_pdf(x, mu=0, sigma=1):
    return(math.exp(-(x-mu)**2)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)
xs_1 = [x/10 for x in range(-50, 50)]
np.random.shuffle(xs_1)
xs_1 = np.sort(xs_1)
# xs_1 = np.sort(xs_)
plt.plot(xs_1, [normal_pdf(x) for x in xs_1])
plt.show()
