import os
import keras.activations
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import scipy.stats as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def sqrt_sigmoid(x):
    # return tf.math.sqrt(tf.sigmoid(x)) * [1, 10, 10, 10, 5, 10, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # return tf.math.sqrt(tf.sigmoid(x)) * [1, 10, 10, 10, 5, 10, 5, 1, 1, 1, 1, 1, 1, 1]
    # return tf.math.sqrt(tf.sigmoid(x)) * [1, 10, 10, 10, 5, 10, 5, 1, 1, 1, 1, 1, 1, 1]
    return tf.math.sqrt(tf.sigmoid(x)) * [1, 10, 10, 10, 5, 10, 5]


class Analyze:
    def __init__(self, data_path):
        self.data_path = data_path
        self.check_alpha = False
        self.data_plot = False
        self.gt_sigma_plot = False
        self.weight_plot = False
        self.fitting_plot = False

        self.error_channel_composition = {"e_y": 1, "e_x": 1, "e_z": 1, "e_h": 1, "e_w": 1, "e_l": 1, "e_theta": 1}
        self.data_channel_composition = {"y": 1, "x": 1, "z": 1, "h": 1, "w": 1, "l": 1, "theta": 1, "object_probs": 1,
                                        }

        # self.error_channel_composition = {"e_y": 1, "e_x": 1, "e_z": 1, "e_h": 1, "e_w": 1, "e_l": 1, "e_theta": 1,
        #                                   "e_object_probs": 1, "e_category": 1,
        #                                   "e_category_probs_0": 1, "e_category_probs_1": 1, "e_category_probs_2": 1,
        #                                   "e_category_probs_3": 1}
        # self.data_channel_composition = {"y": 1, "x": 1, "z": 1, "h": 1, "w": 1, "l": 1, "theta": 1, "object_probs": 1,
        #                                  "category": 1,
        #                                 "category_probs_0": 1, "category_probs_1": 1, "category_probs_2": 1,
        #                                 "category_probs_3": 1
        #                                 }

        # self.error_channel_composition = {"e_y": 1, "e_x": 1, "e_z": 1, "e_h": 1, "e_w": 1, "e_l": 1, "e_theta": 1,
        #                                   "e_object_probs": 1, "e_category": 1,
        #                                   "e_category_probs_0": 1, "e_category_probs_1": 1, "e_category_probs_2": 1,
        #                                   "e_category_probs_3": 1, "e_occlusion_ratio": 1}
        # self.data_channel_composition = {"y": 1, "x": 1, "z": 1, "h": 1, "w": 1, "l": 1, "theta": 1, "object_probs": 1,
        #                                  "category": 1,
        #                                 "category_probs_0": 1, "category_probs_1": 1, "category_probs_2": 1,
        #                                 "category_probs_3": 1, "occlusion_ratio": 1
        #                                 }


        # self.error_channel_composition = {"e_y": 1, "e_x": 1, "e_z": 1, "e_h": 1, "e_w": 1, "e_l": 1, "e_theta": 1,
        #                                   "e_occluded": 1,  "e_occluded_probs_0": 1, "e_occluded_probs_1": 1,
        #                                   "e_occluded_probs_2": 1, "e_object_probs": 1, "e_category": 1,
        #                                   "e_category_probs_0": 1, "e_category_probs_1": 1, "e_category_probs_2": 1,
        #                                   "e_category_probs_3": 1}
        # self.data_channel_composition = {"y": 1, "x": 1, "z": 1, "h": 1, "w": 1, "l": 1, "theta": 1,
        #                                  "occluded": 1,  "occluded_probs_0": 1, "occluded_probs_1": 1,
        #                                  "occluded_probs_2": 1, "object_probs": 1, "category": 1,
        #                                  "category_probs_0": 1, "category_probs_1": 1, "category_probs_2": 1,
        #                                  "category_probs_3": 1}

        self.data_columns = ["e_y", "e_x", "e_z", "e_h", "e_w", "e_l", "e_theta"]
        self.color = ["r", "b", "g", "c", "m", "y", "k", "brown", "olive", "purple"]
        self.tile_value = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        self.pf1 = dict()
        # select occlusion level, -1 for all levels
        self.ocl_level = -1
        self.ocl_df = pd.DataFrame()
        self.ocl_data = {"parameter": [], f"Level {self.ocl_level} mean": [], f"Level {self.ocl_level} median": [],
                         "data count": []}

        self.diff_data = {"parameter": [], "Sigma error prob median":[], "Sigma error prob mean":[],
                          "Sigma error ratio median":[], "Sigma error ratio mean": []}

    def get_dataset(self, degree=0, use_scaler=False, only_tp=True, ocl_level=-1.):
        data = pd.read_csv(self.data_path)
        data = data.drop_duplicates(subset=['y', 'x', 'z'], keep='first', ignore_index=True)
        del data["e_object_probs"]
        del data["occluded"]
        del data["occluded_probs_0"]
        del data["occluded_probs_1"]
        del data["occluded_probs_2"]
        del data["e_occluded"]
        del data["e_occluded_probs_0"]
        del data["e_occluded_probs_1"]
        del data["e_occluded_probs_2"]
        del data["category"]
        del data["category_probs_0"]
        del data["category_probs_1"]
        del data["category_probs_2"]
        del data["category_probs_3"]
        del data["e_category"]
        del data["e_category_probs_0"]
        del data["e_category_probs_1"]
        del data["e_category_probs_2"]
        del data["e_category_probs_3"]
        del data["occlusion_ratio"]
        del data["e_occlusion_ratio"]

        if only_tp:
            mask = data["status"].isin(["fp"])
            data = data[~mask]
        outlier_col = [col for col in data.to_dict().keys() if ("e_" in col) and ("_probs" not in col)]
        data = self.remove_out(data, outlier_col)

        if (0 <= ocl_level) and (ocl_level < 3):
            print("set occluded data")
            data = data[data["occluded"]==ocl_level]

        feature_data = data.loc[:, :"|"].drop(columns=["|"])
        label_data = data.loc[:, "|":].drop(columns=["|", "status"])
        train_feature = feature_data.sample(frac=0.65, random_state=14)
        test_feature = feature_data.drop(train_feature.index)

        train_label = label_data.sample(frac=0.65, random_state=14)
        test_label = label_data.drop(train_label.index)

        # NOT effective
        if use_scaler:
            scaler = StandardScaler()
            train_feature = scaler.fit_transform(train_feature)
            train_feature = pd.DataFrame(train_feature, columns=self.data_channel_composition.keys())
            test_feature = scaler.fit_transform(test_feature)
            test_feature = pd.DataFrame(test_feature, columns=self.data_channel_composition.keys())
        # NOT effective
        if degree > 1:
            poly = PolynomialFeatures(degree=degree)
            train_feature = poly.fit_transform(train_feature)
            test_feature = poly.fit_transform(test_feature)
        return train_feature.values, train_label, test_feature.values, test_label

    def remove_out(self, dataframe, remove_col):
        dff = dataframe
        for k in remove_col:
            level_1q = dff[k].quantile(0.25)
            level_3q = dff[k].quantile(0.75)
            IQR = level_3q - level_1q
            rev_range = 1.5 # 제거 범위 조절 변수
            dff = dff[(dff[k] <= level_3q + (rev_range * IQR)) & (dff[k] >= level_1q - (rev_range * IQR))]
            dff = dff.reset_index(drop=True)
        return dff

    def train_main(self):
        train_data, train_label, test_data, test_label = self.get_dataset(degree=1, use_scaler=True, only_tp=False,
                                                                          ocl_level=self.ocl_level)
        if self.check_alpha:
            self.set_alpha(train_data, train_label, test_data, test_label)

        model = self.set_model("tf_regression")
        model.fit(x=train_data, y=train_label, epochs=1000, batch_size=512, shuffle=False, verbose=1)
        model.save("model.h5")
        print("=========model save=========")

    def set_alpha(self, train_feature, train_label, test_feature, test_label):
        train_score = []
        test_score = []
        alpha_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        # alpha_list = [0.01, 0.1, 1, 10]
        for alpha in alpha_list:
            # model = Ridge(alpha=alpha)
            model = Lasso(alpha=alpha, max_iter=10000)
            # model = ElasticNet(alpha=alpha, l1_ratio=0.0001, max_iter=10000)

            model.fit(train_feature, train_label)
            train_score.append(model.score(train_feature, train_label))
            test_score.append(model.score(test_feature, test_label))

        plt.plot(np.log10(alpha_list), train_score)
        plt.plot(np.log10(alpha_list), test_score)
        plt.xlabel('alpha')
        plt.ylabel('R^2')
        plt.show()

    def set_model(self, model_name="linear"):
        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "ransac":
            model = RANSACRegressor()
        elif model_name == "ridge":
            model = Ridge(alpha=0.00001, max_iter=10000)
        elif model_name == "lasso":
            model = Lasso(alpha=10, max_iter=10000)
        elif model_name == "elastic":
            model = ElasticNet(alpha=10, l1_ratio=0.5, max_iter=10000)
        elif model_name == "tree":
            model = DecisionTreeRegressor()
        elif model_name == "random":
            model = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
        elif model_name == "neighbor":
            model = KNeighborsRegressor(5)
        elif model_name == "svm":
            model = SVR()
        elif model_name == "tf_regression":
            tf.keras.utils.get_custom_objects().update({"sqrt_sigmoid": sqrt_sigmoid})
            model = keras.Sequential()
            model.add(keras.layers.Dense(64, input_dim=8))
            model.add(keras.layers.Activation('softmax'))
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.Activation('softmax'))
            model.add(keras.layers.Dense(7))
            model.add(keras.layers.Activation('sqrt_sigmoid'))
            model.compile(
                loss=tf.keras.losses.Huber(delta=5),  # <-------- Here we define the loss function
                # loss='mse',  # <-------- Here we define the loss function
                #           optimizer=tf.keras.optimizers.RMSprop(0.01),
                #           optimizer=tf.keras.optimizers.SGD(0.01),
                          optimizer=tf.keras.optimizers.Adam(0.01),
                          metrics=['mae', 'mse'])

        return model

    def test_main(self):
        grtr_sigma_train = dict()
        sample_values_train = dict()
        pred_sigma_train_corrected = dict()
        grtr_sigma_test = dict()
        sample_values_test = dict()
        pred_sigma_test_corrected = dict()

        train_data, train_label, test_data, test_label = self.get_dataset(degree=1, use_scaler=True, only_tp=False,
                                                                          ocl_level=self.ocl_level)
        print("=========model load=========")
        model = keras.models.load_model("model.h5", custom_objects={"sqrt_sigmoid": sqrt_sigmoid})
        pred_train = model.predict(train_data)
        pred_val = model.predict(test_data)

        pred_val = self.slice_pred(pred_val, True)
        pred_train = self.slice_pred(pred_train, True)

        if self.data_plot:
            self.raw_data_scatter(pred_train, train_label, "train")
            self.raw_data_scatter(pred_val, test_label, "val")

        for i, key in enumerate(self.data_columns):
            title = key.lstrip("e_")
            train_error_data = train_label[key].to_numpy()
            val_error_data = test_label[key].to_numpy()
            pred_train_data = pred_train[key].flatten()
            pred_val_data = pred_val[key].flatten()

            if (0 <= self.ocl_level) and (self.ocl_level < 3):
                # self.occlusion_data(pred_train_data, key, "train")
                self.occlusion_data(pred_val_data, key, "val")

            # if self.data_plot:
            #     self.raw_data_scatter(pred_train_data, train_error_data, title, i, "train")
            #     self.raw_data_scatter(pred_val_data, val_error_data, title, i, "val")

            grtr_sigma_train[key], sample_values_train[key] = self.create_gt_sigma(pred_train_data, train_error_data, title, "train")
            grtr_sigma_test[key], sample_values_test[key] = self.create_gt_sigma(pred_val_data, val_error_data, title, "val")

            self.correction_const(sample_values_train[key], grtr_sigma_train[key], key)
            pred_sigma_train_corrected[key] = self.data_correction(sample_values_train[key], grtr_sigma_train[key], key)
            pred_sigma_test_corrected[key] = self.data_correction(sample_values_test[key], grtr_sigma_test[key], key)
            # if self.fitting_plot:
            #     self.draw_fit(grtr_sigma_train, sample_values_train, pred_sigma_train_corrected, key, title, "train")
            #     self.draw_fit(grtr_sigma_test, sample_values_test, pred_sigma_test_corrected, key, title, "val")
            # self.draw_sigma(grtr_sigma_test, sample_values_test, pred_sigma_test_corrected, title)
            self.create_dataframe(grtr_sigma_test[key], sample_values_test[key], key)
        if self.fitting_plot:
            self.draw_fit(grtr_sigma_train, sample_values_train, pred_sigma_train_corrected, "train")
            self.draw_fit(grtr_sigma_test, sample_values_test, pred_sigma_test_corrected, "val")
        # self.draw_sigma(grtr_sigma_test, sample_values_test, pred_sigma_test_corrected)
        self.save_csv()

    def slice_pred(self, value, is_error, is_tf=False):
        if is_tf:
            names, channels = list(self.error_channel_composition.keys()), list(self.error_channel_composition.values())
            value = tf.split(value, channels, axis=-1)
            value = dict(zip(names, value))
        else:
            if is_error:
                names, channels = list(self.error_channel_composition.keys()), list(self.error_channel_composition.values())
            else:
                names, channels = list(self.data_channel_composition.keys()), list(self.data_channel_composition.values())
            value = np.split(value, np.sum(channels), axis=1)
            value = dict(zip(names, value))
        return value

    def create_gt_sigma(self, pred, error, title, is_train):
        grtr_sigma = []
        weights = []
        min_value = np.quantile(pred, 0.1)
        max_value = np.quantile(pred, 0.9)
        sample_values = np.linspace(min_value, max_value, 9)
        weight_sigma = ((sample_values[1] - sample_values[0]) / 4)
        # weight_sigma = (sample_values[1] - sample_values[0]) / 16

        for sample_value, color, sample_quantile in zip(sample_values, self.color, self.tile_value):
            sigma_weight = self.weight_norm(pred, sample_value, weight_sigma, sample_quantile)
            weights.append(sigma_weight)
            sigma_gt = np.sqrt(np.sum(sigma_weight * error ** 2) / np.sum(sigma_weight))
            grtr_sigma.append(sigma_gt)
            if self.gt_sigma_plot:
                self.draw_sigma_gt(sample_value, sigma_gt, sample_quantile, color, title, is_train)
        plt.clf()
        if self.weight_plot:
            # self.draw_weight(pred, weights, title)
            self.draw_weight(pred, error, is_train, title)
        return grtr_sigma, sample_values


    def occlusion_data(self, pred, key, is_train):
        pred_upper_value = np.quantile(pred, 0.95)
        ocl_data = pred[pred < pred_upper_value]
        ocl_mean = np.mean(ocl_data)
        ocl_median = np.median(ocl_data)
        self.ocl_data["parameter"].append(key)
        self.ocl_data[f"Level {self.ocl_level} mean"].append(ocl_mean)
        self.ocl_data[f"Level {self.ocl_level} median"].append(ocl_median)
        self.ocl_data["data count"].append(len(ocl_data))

    def weight_norm(self, pred_test, mean_value, weight_sigma, sample_quantile, draw=False, color=None, key=None):
        weight = st.norm.pdf(pred_test, loc=mean_value, scale=weight_sigma)
        return weight

    def correction_const(self, sample_values, grtr_sigma, key):
        self.pf1[key] = np.polyfit(sample_values, grtr_sigma, 1)

    def data_correction(self, sample_values, grtr_sigma, key):
        fitting_data = np.poly1d(self.pf1[key])(sample_values)
        return fitting_data

    def raw_data_scatter(self, data, label, is_train):
        path = f"./raw_data_{is_train}"
        self.make_dir(path)
        plt.figure(figsize=(10, 10))
        for i, key in enumerate(self.data_columns):
            title = key.lstrip("e_")
            error_data = label[key].to_numpy()[..., np.newaxis][data[key] < np.quantile(data[key].flatten(), 0.95)]
            pred_data = data[key][data[key] < np.quantile(data[key].flatten(), 0.95)]
            # error_data = label[key].to_numpy()
            # pred_data = data[key].flatten()
            if title == "theta":
                i = 7
            # plt.rc("font", size=5)
            plt.subplot(3, 3, i + 1)
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.scatter(pred_data, error_data, s=1)
            plt.title(f"{title}")
            plt.ylabel(f"error {title}")
            plt.xlabel(f"predicted sigma {title}")
        # plt.show()
        plt.savefig(f"{path}/raw_data_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png", dpi=600,
                    bbox_inches='tight')
        plt.clf()

    def make_dir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def draw_sigma_gt(self, sample_value, sigma_gt, sample_quantile, color, title, is_train):
        path = f"./sigma_gt_data_{is_train}"
        self.make_dir(path)
        plt.scatter(sample_value, sigma_gt, label=f"{title}_{sample_quantile}", s=50, c=color)
        plt.title(title)
        plt.xlabel("pred")
        plt.ylabel("gt")
        plt.savefig(f"{path}/sigma_gt_{title}_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png",
                    dpi=600)

    def draw_weight(self, pred, error, is_train, title):
        path = f"./weight"
        self.make_dir(path)
        sorted_pred = np.sort(pred)
        min_value = np.quantile(sorted_pred, 0.1)
        max_value = np.quantile(sorted_pred, 0.9)
        sample_values = np.linspace(min_value, max_value, 9)
        weight_sigma = (sample_values[1] - sample_values[0]) / 4
        for sample_value, color, sample_quantile in zip(sample_values, self.color, self.tile_value):
            sigma_weight = self.weight_norm(sorted_pred, sample_value, weight_sigma, sample_quantile)
            cut_pred = sorted_pred[sorted_pred < sample_value + weight_sigma*4]
            tcut_pred = cut_pred[cut_pred > sample_value - weight_sigma*4]
            # plt.plot(cut_pred)
            # plt.show()
            cut_sigma_weight = sigma_weight[sorted_pred < sample_value + weight_sigma*4]
            cut_sigma_weight = cut_sigma_weight[cut_pred > sample_value - weight_sigma*4]
            # plt.plot(tcut_pred, cut_sigma_weight)
            # plt.show()
            plt.figure(figsize=(10,5))
            plt.plot(tcut_pred, cut_sigma_weight*0.3)
            plt.scatter(pred, error, s=10, color='red')
            # plt.show()
            plt.xlim([np.min(sorted_pred), 2])
            plt.xlabel("pred standard deviation")
            plt.ylabel("error")
            plt.savefig(f"{path}/weight_{sample_quantile}_{title}_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png",
                        bbox_inches='tight', dpi=600)
            # plt.xlabel("pred")
            # plt.ylabel("weight")
            plt.clf()
        # plt.scatter(pred, error, s=10)
        # plt.title(title)
        # plt.xlabel("pred")
        # plt.ylabel("weight")
        # plt.show()
        # plt.savefig(f"{path}/weight_{title}_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png",
        #             dpi=600, bbox_inches='tight')
        # plt.clf()

    def draw_fit(self, grtr_sigma, sample_values, corretion_data, is_train):
        path = f"./correction_data"
        self.make_dir(path)
        plt.figure(figsize=(10, 10))
        for i, key in enumerate(self.data_columns):
            title = key.lstrip("e_")
            if title == "theta":
                i = 7
                title = "θ"
            plt.subplot(3, 3, i+1)
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
            plt.plot(sample_values[key], np.poly1d(self.pf1[key])(sample_values[key]), color="blue")
            plt.scatter(sample_values[key], corretion_data[key], s=50, c="olive", label="pred")
            plt.scatter(sample_values[key], grtr_sigma[key], s=50, c='red')
            plt.ylim([0, np.max(grtr_sigma[key])+np.max(grtr_sigma[key])*0.25])
            plt.title(title)
            plt.xlabel(f"predicted std {title}")
            plt.ylabel(f"std {title}")
        # plt.show()
        plt.savefig(f"{path}/corrected_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png", dpi=600,
                    bbox_inches='tight')
        plt.clf()

    def draw_sigma(self, gt_sigma, sample_values, corrected_data, title):
        path = f"./result_plot"
        self.make_dir(path)
        plt.scatter(sample_values, corrected_data, s=10, c="olive")
        plt.scatter(sample_values, gt_sigma, s=50, c="skyblue")
        plt.title(title)
        plt.xlabel("sample value")
        plt.ylabel("sigma value")
        # plt.show()
        plt.savefig(f"{path}/corrected_{title}.png", facecolor="#eeeeee", edgecolor="black", format="png", dpi=600)
        plt.clf()

    def create_dataframe(self, grtr_sigma, sample_values, key):
        diff = (np.abs(np.poly1d(self.pf1[key])(sample_values) - grtr_sigma))
        diff_ratio = (np.abs(np.poly1d(self.pf1[key])(sample_values) - grtr_sigma) / grtr_sigma)
        self.diff_data["parameter"].append(key)
        self.diff_data["Sigma error prob median"].append(f"{np.median(diff):.4f}")
        self.diff_data["Sigma error prob mean"].append(f"{np.mean(diff):.4f}")
        self.diff_data["Sigma error ratio median"].append(f"{np.median(diff_ratio) * 100:.2f}")
        self.diff_data["Sigma error ratio mean"].append(f"{np.mean(diff_ratio) * 100:.2f}")

    def save_csv(self):
        diff_df = pd.DataFrame(self.diff_data)
        diff_df.to_csv("diff_table.csv", index=False, float_format="%.4f")

        if (0 <= self.ocl_level) and (self.ocl_level < 3):
            ocl_df = pd.DataFrame(self.ocl_data)
            ocl_df.to_csv("ocl_table.csv", index=False, float_format="%.4f")

def main():
    # analyze = Analyze("/home/eagle/mun_workspace/ckpt/test_1202_v1/anaylze/analyze_data_(copy).csv")
    analyze = Analyze("/home/eagle/mun_workspace/ckpt/yolov3_3d/anaylze/analyze_data.csv")
    print("=========start train=========")
    analyze.train_main()
    print("=========start test=========")
    analyze.test_main()


if __name__ == "__main__":
    main()