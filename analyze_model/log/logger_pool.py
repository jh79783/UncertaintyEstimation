import numpy as np

import config as cfg
import utils.util_function as uf


class LogBase:
    def __init__(self, logkey):
        self.logkey = logkey

    def compute_mean(self, feature, mask, valid_num):
        if mask is None:
            return np.sum(feature, dtype=np.float32) / valid_num
        return np.sum(feature * mask[..., 0], dtype=np.float32) / valid_num

    def compute_sum(self, feature, mask):
        return np.sum(feature * mask, dtype=np.float32)

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape, dtype=np.float32)[grtr_category[..., 0].astype(np.int32)]
        return one_hot_data


class LogMeanLoss(LogBase):
    def __call__(self, grtr, pred, loss):
        # mean over all feature maps
        return loss[self.logkey]
