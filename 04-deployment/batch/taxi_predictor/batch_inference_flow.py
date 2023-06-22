import argparse
import pandas as pd
import uuid
from taxi_predictor.model import TaxiRidePredictor
from taxi_predictor.utils import upload_to_gcs
from prefect import task, Flow


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


@task
def load_and_clean_data(file):
    return TaxiRidePredictor.load_and_clean_data(file)


@task
def train_model(df_train, df_val, target, params):
    predictor = TaxiRidePredictor(params=params)
    predictor.train(df_train, df_val, target)
    return predictor.model_uri


@task
def predict_duration(model_uri, year, month, taxi_type, model_version, bucket_name):
    predictor = TaxiRidePredictor()
    predictor.load_model(model_uri)

    # ... the rest of the run_prediction code ...

    # Batch prediction on another dataset
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"

    output_df = predictor.process_and_save(input_file, output_file, model_version)

    # Upload to GCS
    blob_name = f"data/predictions/{output_file.split('/')[-1]}"  # Specify your desired blob name in GCS
    upload_to_gcs(output_file, blob_name, bucket_name)

    return output_df


@Flow("TaxiRideDurationPrediction")
def ride_duration_prediction(
    train_file,
    val_file,
    target,
    params,
    year,
    month,
    taxi_type,
    model_version,
    bucket_name,
):
    df_train = load_and_clean_data(train_file)
    df_val = load_and_clean_data(val_file)
    model_uri = train_model(df_train, df_val, target, params)
    results = predict_duration(
        model_uri, year, month, taxi_type, model_version, bucket_name
    )


def run():
    parser = argparse.ArgumentParser(description="Taxi Ride Predictor")
    # ... same argparse setup as in your code ...

    args = parser.parse_args()

    # Setting the params dictionary from argparse arguments
    params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
    }

    ride_duration_prediction.run(
        train_file=args.train_file,
        val_file=args.val_file,
        target=args.target,
        params=params,
        year=args.year,
        month=args.month,
        taxi_type=args.taxi_type,
        model_version=args.model_version,
        bucket_name=args.bucket_name,
    )


if __name__ == "__main__":
    run()
