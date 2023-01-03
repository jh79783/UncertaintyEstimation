import os
import os.path as op
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

import config_dir.util_config as uc
import config as cfg
from dataloader.dataset_reader import DatasetReader
from model.model_factory import ModelFactory
from train.loss_factory import IntegratedLoss
import train.train_val as tv
import train_util as tu


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights):
    batch_size, data_path, ckpt_path = cfg.Train.BATCH_SIZE, cfg.Paths.DATAPATH, \
                                       op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    start_epoch = read_previous_epoch(ckpt_path)
    # dataset_train, train_steps, data_shape,  = get_dataset(data_path, dataset_name, True, batch_size, "train")
    # dataset_val, val_steps, _ = get_dataset(data_path, dataset_name, False, batch_size, "val")

    model, loss_object, optimizer = create_training_parts(batch_size, data_shape, learning_rate, loss_weights,  ckpt_path)
    # model = create_training_parts(batch_size, data_shape, learning_rate, loss_weights,  ckpt_path)
    trainer = tv.ModelTrainer(model, loss_object, optimizer, train_steps, ckpt_path)
    validater = tv.ModelValidater(model, loss_object, val_steps, ckpt_path)
    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        detail_log = (epoch in cfg.Train.DETAIL_LOG_EPOCHS)
        trainer.run_epoch(dataset_train, epoch)
        validater.run_epoch(dataset_val, epoch, detail_log, detail_log)
        save_model_ckpt(ckpt_path, model)
    save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def create_training_parts(batch_size, data_shape, learning_rate, loss_weights, ckpt_path, weight_suffix='latest'):
    model = ModelFactory(batch_size, data_shape).get_model()
    # model = ModelFactory(batch_size, data_shape).build_model()
    model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = IntegratedLoss(loss_weights)
    optimizer = tf.optimizers.Adam(lr=learning_rate)
    return model, loss_object, optimizer


def save_model_ckpt(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if not op.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    print("=== save model:", ckpt_file)
    model.save_weights(ckpt_file)
    # model.save('./my_model')


# def get_dataset(data_path, dataset_name, shuffle, batch_size, split):
#     data_split_path = op.join(data_path, f"{dataset_name}_{split}")
#     reader = DatasetReader(data_split_path, shuffle, batch_size, 1)
#     dataset = reader.get_dataset()
#     frames = reader.get_total_frames()
#     dataset_cfg = reader.get_dataset_config()
#     input_shape = dataset_cfg["data"]["shape"]
#     return dataset, frames // batch_size, input_shape



def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model = tu.load_weights(model, ckpt_file)
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def read_previous_epoch(ckpt_path):
    filename = op.join(ckpt_path, 'history.csv')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            print("[read_previous_epoch] EMPTY history:", history)
            return 0

        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        print(f"[read_previous_epoch] NO history in {filename}")
        return 0

