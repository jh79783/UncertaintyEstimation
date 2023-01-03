import numpy as np

import config_dir.config_generator as cg
import config as cfg


def get_channel_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_MAIN_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_MAIN_COMPOSITION


def get_bbox_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_NMS_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_NMS_COMPOSITION


def get_valid_category_mask(dataset="kitti"):
    """
    :param dataset: dataset name
    :return: binary mask e.g. when
        Dataloader.MAJOR_CATE = ["Person", "Car", "Van", "Bicycle"] and
        Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
        Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
        this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
        but "Bicycle" is not
    """
    dataset_cfg = cg.set_dataset_and_get_config(dataset)
    renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                          for categ in dataset_cfg.CATEGORIES_TO_USE]
    if dataset == "kitti":
        for i, categ in enumerate(cfg.Dataloader.CATEGORY_NAMES["category"]):
            if categ not in renamed_categories:
                renamed_categories.insert(i, categ)

    mask = np.zeros((len(cfg.Dataloader.CATEGORY_NAMES["category"]),), dtype=np.int32)
    for categ in renamed_categories:
        if categ in cfg.Dataloader.CATEGORY_NAMES["category"]:
            index = cfg.Dataloader.CATEGORY_NAMES["category"].index(categ)
            if index < len(cfg.Dataloader.CATEGORY_NAMES["category"]):
                mask[index] = 1
    return mask