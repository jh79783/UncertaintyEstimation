import tensorflow as tf
import config_dir.util_config as uc


class FeatureDecoder:
    def __init__(self, channel_compos=uc.get_channel_composition(False)):
        self.channel_compos = channel_compos

    def decode(self, feature):
        decoded = dict()
        for key in feature.keys():
            if key is not "whole":
                decoded[key] = tf.sigmoid(feature[key])

        pred = [decoded[key] for key in self.channel_compos]
        decoded["whole"] = (tf.concat(pred, axis=-1))
        assert decoded["whole"].shape == feature["whole"].shape
        return decoded
