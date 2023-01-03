import numpy as np

import dataloader.preprocess as pr
import config as cfg


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg,
                 category_names=cfg.Dataloader.CATEGORY_NAMES,
                 max_data=cfg.Dataloader.MAX_DATA):
        self.data_reader = data_reader
        self.category_names = category_names
        self.max_data = max_data
        self.preprocess_example = pr.ExamplePreprocess(max_data=max_data)

    def get_example(self, index):
        example = dict()
        example["data"] = self.data_reader.get_bboxes(index)
        example["box_error"] = self.data_reader.get_error(index)
        example = self.preprocess_example(example)
        return example

    def merge_box_and_category(self, bboxes, categories):
        reamapped_categories = []
        for category_str in categories:
            if category_str in self.category_names["major"]:
                major_index = self.category_names["major"].index(category_str)
                minor_index = -1
                speed_index = -1
            elif category_str in self.category_names["sign"]:
                major_index = self.category_names["major"].index("Traffic sign")
                minor_index = self.category_names["sign"].index(category_str)
                speed_index = -1
            elif category_str in self.category_names["mark"]:
                major_index = self.category_names["major"].index("Road mark")
                minor_index = self.category_names["mark"].index(category_str)
                speed_index = -1
            elif category_str in self.category_names["sign_speed"]:
                major_index = self.category_names["major"].index("Traffic sign")
                minor_index = self.category_names["sign"].index("TS_SPEED_LIMIT")
                speed_index = self.category_names["sign_speed"].index(category_str)
            elif category_str in self.category_names["mark_speed"]:
                major_index = self.category_names["major"].index("Road mark")
                minor_index = self.category_names["mark"].index("RM_SPEED_LIMIT")
                speed_index = self.category_names["mark_speed"].index(category_str)
            elif category_str in self.category_names["dont"]:
                major_index = -1
                minor_index = -1
                speed_index = -1
            else:
                major_index = -2
                minor_index = -2
                speed_index = -2
            reamapped_categories.append((major_index, minor_index, speed_index))
        reamapped_categories = np.array(reamapped_categories)
        # bbox: yxhw, obj, major_ctgr, minor_ctgr, speed_index, depth (9)
        bboxes = np.concatenate([bboxes[..., :-1], reamapped_categories, bboxes[..., -1:]], axis=-1)
        dontcare = bboxes[bboxes[..., 5] == -1]
        bboxes = bboxes[bboxes[..., 5] >= 0]
        return bboxes, dontcare
