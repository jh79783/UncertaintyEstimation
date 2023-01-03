import tensorflow as tf

import config as cfg


def do_nothing(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


mode_decor = None
if cfg.Train.MODE in ["graph", "distribute"]:
    mode_decor = tf.function
else:
    mode_decor = do_nothing


def load_weights(model, ckpt_file):
    model.load_weights(ckpt_file)
    return model
