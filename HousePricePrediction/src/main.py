import argparse

from housePrediction import ingest_data, score, train


def model_result():

    data_path = "./datasets/housing/housing.csv"

    # Get data
    ingest_data_class = ingest_data.IngestData()
    (
        housing_prepared,
        test_data,
        housing_labels,
    ) = ingest_data_class.prepare_dataset()

    # Train model and fit
    model_detail = train.TrainModel(
        housing_prepared,
        housing_labels,
        model_output_folder="artifacts/",
    )
    model_linear = model_detail.linear_model()

    # Score model
    scores = score.CalculateScore(model="linear", metric="mae")
    results = scores.calculate_score()

    return results


if __name__ == "__main__":

    # Take hostname and port number from User
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        "-ip",
        help="IP to connect to",
        default="localhost",
    )
    parser.add_argument("--portname", "-p", help="port number", default="5000")
    parser.add_argument(
        "--experiment_name",
        "-e",
        default="House_prediction",
        help="Experiment name",
    )

    args = parser.parse_args()

    print(model_result())
