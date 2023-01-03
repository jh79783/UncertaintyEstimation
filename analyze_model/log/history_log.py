import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.logger_pool import LogMeanLoss


class HistoryLog:
    def __init__(self, loss_names):
        self.columns = loss_names + cfg.Log.HistoryLog.SUMMARY
        self.loggers = self.create_loggers(self.columns)
        self.data = pd.DataFrame()
        self.summary = dict()

    def __call__(self, step, grtr, pred, loss, total_loss):
        result = dict()
        for key, log_object in self.loggers.items():
            result[key] = log_object(grtr, pred, loss)
        result.update({"total_loss": total_loss.numpy()})
        self.data = self.data.append(result, ignore_index=True)

    def create_loggers(self, columns):
        loggers = dict()
        if "mse" in columns:
            loggers["mse"] = LogMeanLoss("mse")
        if "smooth_l1" in columns:
            loggers["smooth_l1"] = LogMeanLoss("smooth_l1")
        if "category_2d" in columns:
            loggers["category_2d"] = LogMeanLoss("category_2d")
        if "category_3d" in columns:
            loggers["category_3d"] = LogMeanLoss("category_3d")
        if "yx" in columns:
            loggers["yx"] = LogMeanLoss("yx")
        if "hwl" in columns:
            loggers["hwl"] = LogMeanLoss("hwl")
        if "depth" in columns:
            loggers["depth"] = LogMeanLoss("depth")
        if "theta" in columns:
            loggers["theta"] = LogMeanLoss("theta")
        return loggers

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        self.make_summary(epoch_time)
        # make summary dataframe

    def make_summary(self, epoch_time):
        mean_result = self.data.mean(axis=0).to_dict()
        sum_result = self.data.sum(axis=0).to_dict()
        sum_result = {"recall": sum_result["trpo"] / (sum_result["grtr"] + 1e-5),
                      "precision": sum_result["trpo"] / (sum_result["pred"] + 1e-5)}
        metric_keys = ["trpo", "grtr", "pred"]
        # sum_result = {"recall": sum_result["trpo"] / (sum_result["grtr"] + 1e-5),
        #               "precision": sum_result["trpo"] / (sum_result["pred"] + 1e-5),
        #               "recall3d": sum_result["trpo3d"] / (sum_result["grtr3d"] + 1e-5),
        #               "precision3d": sum_result["trpo3d"] / (sum_result["pred3d"] + 1e-5)
        #               }
        # metric_keys = ["trpo", "grtr", "pred", "trpo3d", "grtr3d", "pred3d"]
        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(sum_result)
        summary["time_m"] = round(epoch_time, 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary


