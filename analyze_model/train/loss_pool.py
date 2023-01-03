import tensorflow as tf


class LossBase:
    def __call__(self, grtr, pred):
        dummy_large = tf.reduce_mean(tf.square(pred["feature_l"]))
        dummy_medium = tf.reduce_mean(tf.square(pred["feature_m"]))
        dummy_small = tf.reduce_mean(tf.square(pred["feature_s"]))
        dummy_total = dummy_large + dummy_medium + dummy_small
        return dummy_total, {"dummy_large": dummy_large, "dummy_medium": dummy_medium, "dummy_small": dummy_small}


class MSELoss(LossBase):
    def __call__(self, grtr, pred):
        loss = tf.reduce_mean(tf.square(grtr["box_error"] - pred["box_error"]))
        return loss


class SmoothL1Loss(LossBase):
    def __call__(self, grtr, pred):
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        loss = tf.reduce_sum(huber_loss(grtr, pred))
        return loss
