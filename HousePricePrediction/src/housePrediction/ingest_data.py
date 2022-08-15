import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housePrediction.helper import fetch_housing_data
from housePrediction.transformer import CombinedAttributesAdder

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


class IngestData:
    def __init__(self, output_path=HOUSING_PATH):
        self.output_path = output_path
        self.stratified_shuffle = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )

        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")

        self.combined_attr_adder = CombinedAttributesAdder()

    def load_housing_data(self):
        logging.info("Loading housing data.")
        if not os.path.exists("./datasets/housing/housing.csv"):
            fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
        csv_path = os.path.join(self.output_path, "housing.csv")
        return pd.read_csv(csv_path)

    def prepare_dataset(self):
        logging.info("Preparing datasets.")
        housing = self.load_housing_data()

        # Split data to get train and test set
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )
        for train_index, test_index in self.stratified_shuffle.split(
            housing, housing["income_cat"]
        ):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Now you should remove the income_cat attribute
        # so the data is back to its original state:

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        # Seperate X and y from Train set

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        # Cleaning data
        cat_column = ["ocean_proximity"]
        num_column = housing.drop("ocean_proximity", axis=1).columns

        num_pipe = Pipeline(
            steps=[
                ("imputer", self.num_imputer),
                ("attribs_adder", self.combined_attr_adder),
                ("std_scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", self.cat_imputer),
                ("one_hot_encoder", OneHotEncoder()),
            ]
        )

        full_pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_column),
                ("cat", cat_pipe, cat_column),
            ]
        )

        # Pipeline results
        housing_prepared = full_pipeline.fit_transform(housing)
        test_data = full_pipeline.transform(strat_test_set)

        # Save results
        np.save("datasets/housing_prepared.npy", housing_prepared)
        np.save("datasets/test_data.npy", test_data)

        housing_labels.to_csv("datasets/housing_labels.csv", index=False)
        logging.info("Dataset prepared.")

        return (
            housing_prepared,
            test_data,
            housing_labels,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        "-o",
        help="Output folder/file path",
        default=HOUSING_PATH,
    )
    args = parser.parse_args()
    ingest = IngestData(args.output_path)
    ingest.prepare_dataset()
