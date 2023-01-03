from train.loss_pool import *
import utils.util_function as uf
import config as cfg


class IntegratedLoss:
    def __init__(self, loss_weights):
        self.loss_names = [key for key in loss_weights.keys()]
        self.loss_weights = loss_weights
        self.scaled_loss_objects = self.create_scale_loss_objects(loss_weights)

    def create_scale_loss_objects(self, loss_weights):
        loss_objects = dict()
        for loss_name, values in loss_weights.items():
            loss_objects[loss_name] = eval(values[1])(*values[2:])
        return loss_objects

    def __call__(self, features, predictions):
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_weights}

