import argparse

import mlflow
import mlflow.sklearn

from housePrediction import ingest_data, score, train


def model_result():

    data_path = "./datasets/housing/housing.csv"

    with mlflow.start_run(run_name="MODELLING_RUN"):

        with mlflow.start_run(run_name="DATA_INGESTION_RUN", nested=True):

            # Get data
            ingest_data_class = ingest_data.IngestData()
            (
                housing_prepared,
                test_data,
                housing_labels,
            ) = ingest_data_class.prepare_dataset()
            mlflow.log_artifacts("datasets", artifact_path="states")

        with mlflow.start_run(run_name="CREATE_MODEL_RUN", nested=True):

            # Train model and fit
            model_detail = train.TrainModel(
                housing_prepared,
                housing_labels,
                model_output_folder="artifacts/",
            )
            model_linear = model_detail.linear_model()
            mlflow.sklearn.log_model(model_linear, "linear_model")

        with mlflow.start_run(run_name="SCORE_RUN", nested=True):

            # Score model
            scores = score.CalculateScore(model="linear", metric="mae")
            results = scores.calculate_score()
            mlflow.log_artifact(data_path)
            mlflow.log_metrics(results)

    return results


if __name__ == "__main__":

    # Take hostname and port number from User
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        "-ip",
        help="IP to connect to",
        default="127.0.0.1",
    )
    parser.add_argument("--portname", "-p", help="port number", default="5000")
    parser.add_argument(
        "--experiment_name",
        "-e",
        default="House_prediction",
        help="Experiment name",
    )

    args = parser.parse_args()

    # Connect to uri
    remote_server_uri = f"http://{args.hostname}:{args.portname}"
    mlflow.set_tracking_uri(remote_server_uri)

    # Experiment name
    exp_name = args.experiment_name
    mlflow.set_experiment(exp_name)

    print(model_result())
