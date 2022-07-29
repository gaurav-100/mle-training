import argparse
import logging
import pickle

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeRegressor


class TrainModel:
    def __init__(self, args):
        self.strat_test_set = pd.read_csv(
            args.input_dataset + "strat_test_set.csv"
        )
        self.housing_prepared = pd.read_csv(
            args.input_dataset + "housing_prepared.csv"
        )
        self.housing_labels = pd.read_csv(
            args.input_dataset + "housing_labels.csv"
        )
        self.model_output_folder = args.model_output
        self.lin_reg = LinearRegression()
        self.tree_reg = DecisionTreeRegressor(random_state=42)
        self.forest_reg = RandomForestRegressor(random_state=42)
        self.param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }
        self.param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {
                "n_estimators": [3, 10, 30],
                "max_features": [2, 4, 6, 8],
            },
            # then try 6 (2×3) combinations with bootstrap set as False
            {
                "bootstrap": [False],
                "n_estimators": [3, 10],
                "max_features": [2, 3, 4],
            },
        ]
        self.rnd_search = RandomizedSearchCV(
            self.forest_reg,
            param_distributions=self.param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        self.grid_search = GridSearchCV(
            self.forest_reg,
            self.param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )

    def linear_model(self):

        self.lin_reg.fit(
            self.housing_prepared, self.housing_labels
        )

        with open(
            self.model_output_folder + "linear_model" + ".pickle",
            "wb",
        ) as files:
            pickle.dump(self.lin_reg, files)

    def decision_tree_model(self):

        self.tree_reg.fit(
            self.housing_prepared, self.housing_labels
        )

        with open(
            self.model_output_folder
            + "decision_tree_model"
            + ".pickle",
            "wb",
        ) as files:
            pickle.dump(self.tree_reg, files)

    def grid_search_model(self):

        self.grid_search.fit(
            self.housing_prepared, self.housing_labels
        )

        final_model = self.grid_search.best_estimator_

        with open(
            self.model_output_folder
            + "grid_cv_random_forest"
            + ".pickle",
            "wb",
        ) as files:
            pickle.dump(final_model, files)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output",
        "-o",
        default="artifacts/",
        help="Model output folder",
    )
    parser.add_argument(
        "--input_dataset",
        "-i",
        default="datasets/",
        help="Dataset folder",
    )
    args = parser.parse_args()

    train_model = TrainModel(args)
    logging.info("Training linear model ...")
    train_model.linear_model()
    logging.info("Training decision tree model ...")
    train_model.decision_tree_model()
    logging.info("Grid search ...")
    train_model.grid_search_model()
