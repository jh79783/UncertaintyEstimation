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


class Analyze:
    def __init__(self, data_path):
        self.data_path = data_path
        self.error_channel_composition = {"e_y": 1, "e_x": 1, "e_z": 1, "e_h": 1, "e_w": 1, "e_l": 1, "e_theta": 1,
                                          "e_occluded": 1,  "e_occluded_probs_0": 1, "e_occluded_probs_1": 1,
                                          "e_occluded_probs_2": 1, "e_object_probs": 1, "e_category": 1,
                                          "e_category_probs_0": 1, "e_category_probs_1": 1, "e_category_probs_2": 1,
                                          "e_category_probs_3": 1}
        self.data_channel_composition = {"y": 1, "x": 1, "z": 1, "h": 1, "w": 1, "l": 1, "theta": 1,
                                         "occluded": 1,  "occluded_probs_0": 1, "occluded_probs_1": 1,
                                         "occluded_probs_2": 1, "object_probs": 1, "category": 1,
                                         "category_probs_0": 1, "category_probs_1": 1, "category_probs_2": 1,
                                         "category_probs_3": 1}
        self.data_columns = ["e_y", "e_x", "e_z", "e_h", "e_w", "e_l", "e_theta"]
        self.color = ["r", "b", "g", "c", "m", "y", "k", "brown", "olive", "purple"]
        self.tile_value = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        self.weight_value = {"e_y": 0.015, "e_x": 0.01, "e_z": 0.018, "e_h": 0.01, "e_w": 0.02, "e_l": 0.02, "e_theta": 0.02}

        self.pf1 = dict()
        self.acvitation_wieght = [1, 10, 10, 10, 5, 10, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # select occlusion level, -1 for all levels
        self.ocl_level = -1


    def __call__(self, check_alpha=False):
        train_data, train_label, test_data, test_label = self.get_dataset(degree=1, use_scaler=True, only_tp=False, ocl_level=self.ocl_level)
        if check_alpha:
            self.set_alpha(train_data, train_label, test_data, test_label)

        # model = self.set_model("tf_regression")
        # model.fit(x=train_data, y=train_label, epochs=1000, batch_size=512, shuffle=False, verbose=1)
        # model.save("model.h5")

        model = keras.models.load_model("model.h5", custom_objects={"sqrt_sigmoid": sqrt_sigmoid})

        # print("train score: ", model.score(train_data, train_sigma))
        # print("test score: ", model.score(test_data, test_sigma))
        pred_train = model.predict(train_data.values)
        pred_test = model.predict(test_data.values)

        pred_test = self.slice_pred(pred_test, True)
        pred_train = self.slice_pred(pred_train, True)

        # self.draw_plot(test_data, test_sigma, pred_test)

        self.error_mean(pred_train, train_label, "train", data_plot=False, fitting_plot=False,
                        gt_sigma_plot=False, save_plot=False, )
        self.error_mean(pred_test, test_label, "val", data_plot=True, fitting_plot=False, gt_sigma_plot=False,
                        weight_plot=False, save_plot=True, ocl_level=self.ocl_level)

    def get_dataset(self, degree=0, use_scaler=False, only_tp=True, ocl_level=-1.):
        data = pd.read_csv(self.data_path)
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
        # train_label["e_x"].plot.hist()
        # plt.show()
        return train_feature, train_label, test_feature, test_label

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
            model.add(keras.layers.Dense(64, input_dim=17))
            model.add(keras.layers.Activation('softmax'))
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.Activation('softmax'))
            model.add(keras.layers.Dense(17))
            model.add(keras.layers.Activation('sqrt_sigmoid'))
            model.compile(
                loss=tf.keras.losses.Huber(delta=5),  # <-------- Here we define the loss function
                # loss='mse',  # <-------- Here we define the loss function
                #           optimizer=tf.keras.optimizers.RMSprop(0.01),
                #           optimizer=tf.keras.optimizers.SGD(0.01),
                          optimizer=tf.keras.optimizers.Adam(0.01),
                          metrics=['mae', 'mse'])

        return model

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

    def draw_plot(self, test_data, test_sigma, pred_test):
        plt.subplot(2, 4, 1)
        plt.scatter(pred_test["e_y"], test_sigma["e_y"], c="steelblue", edgecolor="white", s=70)
        plt.title("y")
        plt.xlabel("sigma y")
        plt.ylabel("error y")
        plt.subplot(2, 4, 2)
        plt.scatter(pred_test["e_x"], test_sigma["e_x"], c="g", edgecolor="white", s=70)
        plt.title("x")
        plt.xlabel("sigma x")
        plt.ylabel("error x")
        plt.subplot(2, 4, 3)
        plt.scatter(pred_test["e_z"], test_sigma["e_z"], c="purple", edgecolors="white", s=70)
        plt.title("z")
        plt.xlabel("sigma z")
        plt.ylabel("error z")
        plt.subplot(2, 4, 4)
        plt.scatter(pred_test["e_h"], test_sigma["e_h"], c="m", edgecolor="white", s=70)
        plt.title("h")
        plt.xlabel("sigma h")
        plt.ylabel("error h")
        plt.subplot(2, 4, 5)
        plt.scatter(pred_test["e_w"], test_sigma["e_w"], c="c", edgecolor="white", s=70)
        plt.title("w")
        plt.xlabel("sigma w")
        plt.ylabel("error w")
        plt.subplot(2, 4, 6)
        plt.scatter(pred_test["e_l"], test_sigma["e_l"], c="y", edgecolor="white", s=70)
        plt.title("l")
        plt.xlabel("sigma l")
        plt.ylabel("error l")
        plt.subplot(2, 4, 7)
        plt.scatter(pred_test["e_theta"], test_sigma["e_theta"], c="r", edgecolor="white", s=70)
        plt.title("ry")
        plt.xlabel("sigma ry")
        plt.ylabel("error ry")

        plt.show()

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

    def error_mean(self, pred_test, label_sigma, is_train, data_plot=False, weight_plot=False, gt_sigma_plot=False,
                   fitting_plot=False, save_plot=False, ocl_level=-1.):
        score = {}
        print(is_train)
        for key in self.data_columns:
            title = key.lstrip("e_")
            grtr_sigma = []
            error_data = label_sigma[key].to_numpy()
            pred_data = pred_test[key].flatten()

            weight = np.quantile(pred_data, 0.95)
            min_value = np.quantile(pred_data, 0.1)
            max_value = np.quantile(pred_data, 0.9)
            sample_values = np.linspace(min_value, max_value, 9)
            weight_sigma = (sample_values[1] - sample_values[0]) / 4

            if (0 <= ocl_level) and (ocl_level < 3):
                pred_upper_value = np.quantile(pred_data, 0.95)
                ocl_data = pred_data[pred_data < pred_upper_value]
                ocl_mean = np.mean(ocl_data)
                ocl_var = np.var(ocl_data)
                print(f"ocl lev: {self.ocl_level}, key: {key} mean: {ocl_mean:.4f}, var: {ocl_var:.4f}")

            if data_plot:
                plt.scatter(pred_data, error_data, s=10)
                plt.title(f"{title}")
                plt.ylabel("gt")
                plt.xlabel("pred")
                # plt.savefig(f"data_{title}_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png", dpi=600)
            for sample_value, color, sample_quantile in zip(sample_values, self.color, self.tile_value):
                sigma_weight = self.weight_norm(pred_data, sample_value, weight_sigma, sample_quantile, draw=weight_plot,
                                                color=color, key=key)
                sigma_gt = np.sqrt(np.sum(sigma_weight * error_data**2) / np.sum(sigma_weight))
                if gt_sigma_plot:
                    if is_train == "train":
                        plt.plot(sample_value, sigma_gt, color=color, marker="o", label=f"{key}_{i}")
                    else:
                        plt.scatter(sample_value, sigma_gt, label=f"{title}_{sample_quantile}", s=50, c=color)
                        plt.title(title)
                        plt.xlabel("pred")
                        plt.ylabel("gt")
                        plt.legend(ncol=2)

                grtr_sigma.append(sigma_gt)
                score[key.replace("e_", "score_")] = self.score(pred_data, error_data)
            if is_train == "train":
                pf = np.polyfit(sample_values, grtr_sigma, 1)
                self.pf1[key] = pf
            if fitting_plot:
                plt.plot(sample_values, np.poly1d(self.pf1[key])(sample_values), color="blue")
                fit_data = np.poly1d(self.pf1[key])(sample_values)
                plt.scatter(sample_values, fit_data, s=10, c="olive")
            test = np.poly1d(self.pf1[key])(sample_values)
            diff = (np.abs(np.poly1d(self.pf1[key])(sample_values) - grtr_sigma))
            diff_ratio = (np.abs(np.poly1d(self.pf1[key])(sample_values) - grtr_sigma) / np.poly1d(self.pf1[key])(
                sample_values))
            if save_plot:
                plt.savefig(f"{key}_{is_train}.png", facecolor="#eeeeee", edgecolor="black", format="png", dpi=200)
            plt.clf()
            print(f"{key} diff: {np.mean(diff):.4f}")
            # print(f"{key} diff ratio : {np.mean(diff_ratio):.4f}")
            print(f"{key} diff ratio : ", np.mean(diff_ratio) * 100)
        columns = [key for key in score.keys()]
        score_df = pd.DataFrame([score], columns=columns)
        score_df.to_csv("score.csv", index=False, float_format="%.4f")
        # print(f"pred mean: {pred_mean}")
        # print(f"pred var: {pred_var}")

    def weight_norm(self, pred_test, mean_value, weight_sigma, sample_quantile, draw=False, color=None, key=None):
        weight = st.norm.pdf(pred_test, loc=mean_value, scale=weight_sigma)
        return weight

    def score(self, sigma_pred, error_gt):
        error_norm = st.norm.pdf(error_gt, loc=0, scale=sigma_pred)
        score = np.sum(np.nan_to_num(error_norm)) / len(error_gt)
        return score

    def sigmoid_sqrt(self, x, delta=0, low=0, high=1):
        y = tf.math.sqrt(tf.sigmoid(x)) * self.acvitation_wieght
        return y

    def sigmoid_weight(self, x):
        y = {key: [] for key in x.keys()}
        for key in x.keys():
            y[key] = tf.sigmoid(x[key]) * self.acvitation_wieght[key]
        return y

def sqrt_sigmoid(x):
    return tf.math.sqrt(tf.sigmoid(x)) * [1, 10, 10, 10, 5, 10, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def main():
    Analyze("/home/eagle/mun_workspace/ckpt/test_1202_v1/anaylze/analyze_data_(copy).csv")()


if __name__ == "__main__":
    main()
