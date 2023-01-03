import tensorflow as tf
import numpy as np
from tensorflow import keras

import model.base_model as base
import model.decoder as decoder
import utils.util_function as uf
import config as cfg


class ModelFactory:
    def __init__(self, batch_size, input_shape, model_name=cfg.Architecture.MODEL, training=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.model_name = model_name
        self.training = training

    def get_model(self):
        base_model = base.base_model_factory(self.model_name, self.training)
        input_tensor = tf.keras.layers.Input(shape=self.input_shape, batch_size=self.batch_size)
        base_model_features = base_model(input_tensor)
        output_features = dict()
        output_features["feat_logit"] = uf.merge_and_slice_features(base_model_features, False, "feat")
        output_features["feat"] = decoder.FeatureDecoder().decode(output_features["feat_logit"])
        # elif self.model_name == ""
        analyze_model = tf.keras.Model(inputs=input_tensor, outputs=output_features, name="analyze_model")
        return analyze_model
    # def build_model(self):
    #     input_tensor = tf.keras.layers.Input(shape=self.input_shape)
    #     x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(input_tensor)
    #     x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    #     outputs = tf.keras.layers.Dense(15, name="outputs")(x)
    #     model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
    #     model.compile(optimizer=tf.optimizers.Adam(0.001),
    #                   loss='mse',
    #                   )
    #     return model


def test_model_factory():
    print("======= model test")
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
    y_train = np.array([[10, 1], [15, 2], [20, 2], [30, 3]], dtype=np.float32)
    x_test = np.array([[5, 5], [8, 3], [6, 5], [7, 5]], dtype=np.float32)
    y_test = np.array([[10, 8], [1, 20], [80, 30], [100, 50]], dtype=np.float32)
    model = ModelFactory(1,3).get_model()
    model.summary()
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    history = model.fit(x_train, y_train)
    # lr = model.fit(x_train, y_train)
    # print(lr.score(x_train, y_train))
    # print(lr.score(x_test, y_test))


if __name__ == "__main__":
    uf.set_gpu_configs()
    test_model_factory()
