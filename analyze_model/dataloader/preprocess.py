import numpy as np


class PreprocessBase:
    def __call__(self, example):
        """
        :param example: source example
        :return: preprocessed example
        """
        raise NotImplementedError()


class ExamplePreprocess(PreprocessBase):
    def __init__(self, max_data):
        self.preprocess = [ExampleZeroPadData(max_data)]

    def __call__(self, example):
        for process in self.preprocess:
            example = process(example)
        return example


class ExampleZeroPadData(PreprocessBase):
    def __init__(self, max_data):
        self.max_data = max_data

    def __call__(self, example):
        data = example["data"]
        error = example["box_error"]
        if data.shape[0] < self.max_data:
            new_data = np.zeros((self.max_data, data.shape[1]), dtype=np.float32)
            new_data[:data.shape[0]] = data
            example["data"] = new_data
        if error.shape[0] < self.max_data:
            new_error = np.zeros((self.max_data, error.shape[1]), dtype=np.float32)
            new_error[:data.shape[0]] = error
            example["box_error"] = new_error
        return example
