import os.path as op
import config_dir.parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/home/eagle/mun_workspace"
    DATAPATH = op.join(RESULT_ROOT, "tfrecord")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")
    CONFIG_FILENAME = '/home/eagle/mun_workspace/analyze_model/config.py'
    META_CFG_FILENAME = '/home/eagle/mun_workspace/analyze_model/config_dir/meta_config.py'


class Datasets:
    # specific dataset configs MUST have the same items
    class Analyze:
        NAME = "analyze"
        PATH = "/home/eagle/mun_workspace/ckpt/test_1118_v1/anaylze"
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Cyclist"]
        CATEGORY_REMAP = {}

    DATASET_CONFIG = None
    TARGET_DATASET = "analyze"


class Dataloader:
    DATASETS_FOR_TFRECORD = {
        "analyze": ("train", "val"),
    }
    MAX_DATA = 50

    CATEGORY_NAMES = params.TfrParams.CATEGORY_NAMES
    SHARD_SIZE = 2000
    ANCHORS_PIXEL = None


class Train:
    CKPT_NAME = "regression_test"
    MODE = ["eager", "graph", "distribute"][0]
    BATCH_SIZE = 8
    TRAINING_PLAN = params.TrainingPlan.ANAYLZE
    DETAIL_LOG_EPOCHS = list(range(0, 200, 50))


class ModelOutput:
    GRTR_MAIN_COMPOSITION = {"ctgr_probs": 4, "occlusion_probs": 3, "object": 1,
                             "xy": 2, "hwl": 3, "z": 1, "ry": 1
                             }
    PRED_MAIN_COMPOSITION = {"ctgr_probs": 4, "occlusion_probs": 3, "object": 1,
                             "xy": 2, "hwl": 3, "z": 1, "ry": 1
                             }
    GRTR_NMS_COMPOSITION = {"ctgr_probs": 4, "occlusion_probs": 3, "object": 1,
                            "xy": 2, "hwl": 3, "z": 1, "ry": 1
                            }
    PRED_NMS_COMPOSITION = {"ctgr_probs": 4, "occlusion_probs": 3, "object": 1,
                            "xy": 2, "hwl": 3, "z": 1, "ry": 1
                            }


class Architecture:
    MODEL = "Regression"


class Log:
    VISUAL_HEATMAP = True

    class HistoryLog:
        SUMMARY = []


