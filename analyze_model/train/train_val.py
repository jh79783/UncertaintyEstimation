import time
import cv2
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

import config as cfg
import utils.util_function as uf
from train.train_util import mode_decor
import train.train_util as tu
from log.logger import Logger

import dataloader.data_util as du


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps, ckpt_path):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.is_train = True
        self.batch_size = cfg.Train.BATCH_SIZE

    def run_epoch(self, dataset, epoch=0, visual_log=False, exhaustive_log=False, val_only=False):
        logger = Logger(self.loss_object.loss_names, self.ckpt_path, epoch, self.is_train, val_only)
        epoch_start = timer()
        for step, features in enumerate(dataset):
            start = timer()
            prediction, total_loss, loss_by_type, new_features = self.run_batch(features)
            logger.log_batch_result(step, new_features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={timer() - start:.3f}, "
                              f"total={total_loss:.3f}, "
                              f"box={loss_by_type['iou']:.3f}, "
                              f"object={loss_by_type['object']:.3f}, "
                              f"distance={loss_by_type['distance']:.3f}, "
                              f"category={loss_by_type['category']:.3f}, "

                              )

            if step >= self.epoch_steps:
                break
            # if step >= 10:
            #     break

        print("")
        logger.finalize(epoch_start)

    def run_batch(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps, ckpt_path):
        super().__init__(model, loss_object, optimizer, epoch_steps, ckpt_path)

    def run_batch(self, features):
        return self.run_step(features)

    @mode_decor
    def run_step(self, features):
        with tf.GradientTape() as tape:
            prediction = self.model(features["data"], training=True)
            total_loss, loss_by_type = self.loss_object(features, prediction)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type, features


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path):
        super().__init__(model, loss_object, None, epoch_steps, ckpt_path)
        self.is_train = False

    def run_batch(self, features):
        return self.run_step(features)

    @mode_decor
    def run_step(self, features):
        prediction = self.model(features["data"])
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type, features
