import os
import argparse
import mlflow
from prefect import task, flow, get_run_logger
from taxi_predictor.model import TaxiRidePredictor

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("green-taxi-duration")


@task
def load_and_clean_data_task(file):
    logger = get_run_logger()
    logger.info(f"Loading and cleaning data from file: {file}")
    return TaxiRidePredictor.load_and_clean_data(file)


@task
def train_model_task(df_train, df_val, target, params, push_to_storage):
    logger = get_run_logger()
    logger.info("Training model...")
    predictor = TaxiRidePredictor(params=params)
    predictor.train(df_train, df_val, target, push_to_storage)
    return predictor.model_uri


@flow(name="TaxiRideModelTraining")
def training_flow(train_file, val_file, target, params, push_to_storage):
    df_train = load_and_clean_data_task(train_file)
    df_val = load_and_clean_data_task(val_file)

    model_uri = train_model_task(df_train, df_val, target, params, push_to_storage)

    return model_uri


def run():
    parser = argparse.ArgumentParser(description="Train Taxi Ride Predictor")
    parser.add_argument("--train_file", default="data/green_tripdata_2021-01.parquet")
    parser.add_argument("--val_file", default="data/green_tripdata_2021-02.parquet")
    parser.add_argument("--target", default="duration")
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--push_to_storage", action="store_true")

    args = parser.parse_args()

    # Setting the params dictionary from argparse arguments
    params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
    }

    # Execute the training flow
    training_flow(
        train_file=args.train_file,
        val_file=args.val_file,
        target=args.target,
        params=params,
        push_to_storage=args.push_to_storage,
    )


if __name__ == "__main__":
    run()
