import tensorflow as tf
from tensorflow.keras import layers


def base_model_factory(model_name, training):
    if model_name == "Regression":
        return Regression(training)


class Regression:
    def __init__(self, training):
        self.training = training

    def __call__(self, input_tensor):
        x = layers.Dense(64, activation="relu", name="dense_1")(input_tensor)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(15, name="outputs")(x)
        return outputs

