import argparse
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
)

DOWNLOAD_ROOT = (
    "https://raw.githubusercontent.com/ageron/handson-ml/master/"
)
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


class IngestData:
    def __init__(self, output_path):
        self.output_path = output_path
        self.stratified_shuffle = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        self.imputer = SimpleImputer(strategy="median")

    def fetch_housing_data(self):
        os.makedirs(self.output_path, exist_ok=True)
        tgz_path = os.path.join(self.output_path, "housing.tgz")
        urllib.request.urlretrieve(self.output_path, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.output_path)
        housing_tgz.close()

    def load_housing_data(self):
        csv_path = os.path.join(self.output_path, "housing.csv")
        return pd.read_csv(csv_path)

    def income_cat_proportions(self, data):
        return data["income_cat"].value_counts() / len(data)

    def prepare_dataset(self):
        housing = self.load_housing_data

        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        for (
            train_index,
            test_index,
        ) in self.stratified_shuffle.split(
            housing, housing["income_cat"]
        ):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        train_set, test_set = train_test_split(
            housing, test_size=0.2, random_state=42
        )

        compare_props = pd.DataFrame(
            {
                "Overall": self.income_cat_proportions(housing),
                "Stratified": self.income_cat_proportions(
                    strat_test_set
                ),
                "Random": self.income_cat_proportions(test_set),
            }
        ).sort_index()
        compare_props["Rand. %error"] = (
            100
            * compare_props["Random"]
            / compare_props["Overall"]
            - 100
        )
        compare_props["Strat. %error"] = (
            100
            * compare_props["Stratified"]
            / compare_props["Overall"]
            - 100
        )

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude")
        housing.plot(
            kind="scatter",
            x="longitude",
            y="latitude",
            alpha=0.1,
        )

        corr_matrix = housing.corr()
        corr_matrix["median_house_value"].sort_values(
            ascending=False
        )
        housing["rooms_per_household"] = (
            housing["total_rooms"] / housing["households"]
        )
        housing["bedrooms_per_room"] = (
            housing["total_bedrooms"] / housing["total_rooms"]
        )
        housing["population_per_household"] = (
            housing["population"] / housing["households"]
        )

        housing = strat_train_set.drop(
            "median_house_value", axis=1
        )

        # drop labels for training set
        housing_labels = strat_train_set[
            "median_house_value"
        ].copy()

        housing_num = housing.drop("ocean_proximity", axis=1)

        self.imputer.fit(housing_num)
        X = self.imputer.transform(housing_num)

        housing_tr = pd.DataFrame(
            X,
            columns=housing_num.columns,
            index=housing.index,
        )
        housing_tr["rooms_per_household"] = (
            housing_tr["total_rooms"] / housing_tr["households"]
        )
        housing_tr["bedrooms_per_room"] = (
            housing_tr["total_bedrooms"]
            / housing_tr["total_rooms"]
        )
        housing_tr["population_per_household"] = (
            housing_tr["population"] / housing_tr["households"]
        )

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(
            pd.get_dummies(housing_cat, drop_first=True)
        )

        strat_train_set.to_csv(
            "datasets/strat_train_set.csv", index=False
        )
        strat_test_set.to_csv(
            "datasets/strat_test_set.csv", index=False
        )
        housing_prepared.to_csv(
            "datasets/housing_prepared.csv", index=False
        )
        housing_labels.to_csv(
            "datasets/housing_labels.csv", index=False
        )

        return (
            strat_train_set,
            strat_test_set,
            housing_prepared,
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
