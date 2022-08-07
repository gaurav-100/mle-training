import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


class CalculateScore:
    def __init__(
        self,
        model_folder="artifacts/",
        dataset="datasets/",
        model="linear",
        metric="mae",
    ):
        self.model_folder = model_folder
        self.dataset = dataset
        self.model = model
        self.metric = metric
        self.imputer = SimpleImputer(strategy="median")
        self.strat_test_set = np.load(dataset + "test_data.npy")
        self.housing_prepared = np.load(
            dataset + "housing_prepared.npy"
        )
        self.housing_labels = pd.read_csv(
            dataset + "housing_labels.csv"
        )

    def load_model(self):
        logging.info("Loading model ...")
        model_path = "".join(
            [self.model_folder, self.model + "_model" + ".pickle"]
        )

        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def calculate_prediction(self):

        logging.info("Predicting ...")
        model = self.load_model()
        if self.model != "grid_cv_random_forest.pickle":
            prediction = model.predict(self.housing_prepared)
            label = self.housing_labels
        else:
            X_test = self.strat_test_set.drop(
                "median_house_value", axis=1
            )
            label = self.strat_test_set[
                "median_house_value"
            ].copy()

            X_test_num = X_test.drop("ocean_proximity", axis=1)
            X_test_prepared = self.imputer.transform(X_test_num)
            X_test_prepared = pd.DataFrame(
                X_test_prepared,
                columns=X_test_num.columns,
                index=X_test.index,
            )
            X_test_prepared["rooms_per_household"] = (
                X_test_prepared["total_rooms"]
                / X_test_prepared["households"]
            )
            X_test_prepared["bedrooms_per_room"] = (
                X_test_prepared["total_bedrooms"]
                / X_test_prepared["total_rooms"]
            )
            X_test_prepared["population_per_household"] = (
                X_test_prepared["population"]
                / X_test_prepared["households"]
            )

            X_test_cat = X_test[["ocean_proximity"]]
            X_test_prepared = X_test_prepared.join(
                pd.get_dummies(X_test_cat, drop_first=True)
            )
            prediction = model.predict(X_test_prepared)

        return label, prediction

    def calculate_score(self):
        logging.info("Calculating score ...")
        label, prediction = self.calculate_prediction()
        score = {
            "mae": mean_absolute_error(label, prediction),
            "mse": mean_squared_error(label, prediction),
            "rmse": np.sqrt(
                mean_squared_error(label, prediction)
            ),
        }

        return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_folder",
        "-f",
        default="artifacts/",
        help="Model folder",
    )
    parser.add_argument(
        "--dataset",
        "-i",
        default="datasets/",
        help="Input dataset",
    )
    parser.add_argument(
        "--model", "-m", default="linear", help="Model"
    )
    parser.add_argument(
        "--metric",
        "-s",
        default="mae",
        help="Metric want to calculate either MAE, MSE or RMSE",
    )

    args = parser.parse_args()
    calculate_score = CalculateScore(
        args.model_folder, args.dataset.args.model, args.metric
    )
    score = calculate_score.calculate_score()
    print(
        "{} model {} score: {}".format(
            args.model, args.metric.upper(), score
        )
    )
