import numpy as np
import os.path as op
import pandas as pd
import os

import utils.util_function as uf
from log.history_log import HistoryLog


class Logger:
    def __init__(self, loss_names, ckpt_path, epoch, is_train, val_only):
        self.history_logger = HistoryLog(loss_names)
        self.is_train = is_train
        self.ckpt_path = ckpt_path
        self.epoch = epoch
        self.val_only = val_only

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        self.check_nan(grtr, "grtr")
        self.check_nan(pred, "pred")
        self.check_nan(loss_by_type, "loss")

        # pred["error_data"] = uf.slice_feature()
        for key, feature_slices in grtr.items():
            grtr[key] = uf.convert_tensor_to_numpy(feature_slices)
        for key, feature_slices in pred.items():
            pred[key] = uf.convert_tensor_to_numpy(feature_slices)
        loss_by_type = uf.convert_tensor_to_numpy(loss_by_type)

        if step == 0 and self.epoch == 0:
            structures = {"grtr": grtr, "pred": pred, "loss": loss_by_type}
            self.save_model_structure(structures)

        self.history_logger(step, grtr, pred, loss_by_type, total_loss)

    def check_nan(self, features, feat_name):
        if "merged" not in feat_name:
            valid_result = True
            if isinstance(features, dict):
                for name, value in features.items():
                    self.check_nan(value, f"{feat_name}_{name}")
            elif isinstance(features, list):
                for idx, tensor in enumerate(features):
                    self.check_nan(tensor, f"{feat_name}_{idx}")
            else:
                if features.ndim == 0 and (np.isnan(features) or np.isinf(features) or features > 100000000000):
                    print(f"nan loss: {feat_name}, {features}")
                    valid_result = False
                elif not np.isfinite(features.numpy()).all():
                    print(f"nan {feat_name}:", np.quantile(features.numpy(), np.linspace(0, 1, self.num_channel)))
                    valid_result = False
            assert valid_result

    def save_model_structure(self, structures):
        structure_file = op.join(self.ckpt_path, "structure.md")
        f = open(structure_file, "w")
        for key, structure in structures.items():
            f.write(f"- {key}\n")
            space_count = 1
            self.analyze_structure(structure, f, space_count)
        f.close()

    def analyze_structure(self, data, f, space_count, key=""):
        space = "    " * space_count
        if isinstance(data, list):
            for i, datum in enumerate(data):
                if isinstance(datum, dict):
                    # space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    # space_count -= 1
                elif type(datum) == np.ndarray:
                    f.write(f"{space}- {key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {datum}\n")
                    space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    space_count -= 1
        elif isinstance(data, dict):
            for sub_key, datum in data.items():
                if type(datum) == np.ndarray:
                    f.write(f"{space}- {sub_key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {sub_key}\n")

                space_count += 1
                self.analyze_structure(datum, f, space_count, sub_key)
                space_count -= 1
