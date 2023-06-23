import os
import argparse
from taxi_predictor.model import TaxiRidePredictor
from taxi_predictor.utils import upload_to_gcs
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("green-taxi-duration")


def run_training(
    train_file,
    val_file,
    target,
    params,
    push_to_storage=True,
):
    # Prepare your data
    df_train = TaxiRidePredictor.load_and_clean_data(train_file)
    df_val = TaxiRidePredictor.load_and_clean_data(val_file)

    # Initialize the predictor
    predictor = TaxiRidePredictor(params=params)

    # Train the model
    predictor.train(df_train, df_val, target, push_to_storage)

    # Print the model uri
    print(predictor.model_uri)


def run_prediction(
    predictor,
    year,
    month,
    taxi_type,
    model_version,
    bucket_name,
):
    # Load the model using the model_uri
    predictor.load_model(predictor.model_uri)

    # Single ride prediction
    single_ride = {"PULocationID": "10", "DOLocationID": "50", "trip_distance": 40}
    single_duration = predictor.predict(single_ride)
    print(
        f"The predicted duration for the single ride is: {single_duration.iloc[0,0]} minutes"
    )

    # Batch prediction
    batch_rides = [
        {"PULocationID": "10", "DOLocationID": "50", "trip_distance": 40},
        {"PULocationID": "20", "DOLocationID": "60", "trip_distance": 30},
    ]
    batch_durations = predictor.predict(batch_rides)
    print(
        f"The predicted durations for the batch rides are: \n{batch_durations['predictions']} minutes each"
    )

    # Batch prediction on another dataset
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"

    output_df = predictor.process_and_save(input_file, output_file, model_version)
    print(output_df.head(3))

    # Upload to GCS
    blob_name = f"data/predictions/{output_file.split('/')[-1]}"  # Specify your desired blob name in GCS
    upload_to_gcs(output_file, blob_name, bucket_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Taxi Ride Predictor")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="train")
    parser.add_argument("--train_file", default="data/green_tripdata_2021-01.parquet")
    parser.add_argument("--val_file", default="data/green_tripdata_2021-02.parquet")
    parser.add_argument("--target", default="duration")
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--month", type=int, default=2)
    parser.add_argument("--taxi_type", default="green")
    parser.add_argument("--model_version", default="v1.0.0")
    parser.add_argument("--bucket_name", default="mlops-zoomcamp-bucket")

    args = parser.parse_args()

    # Setting the params dictionary from argparse arguments
    params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
    }

    predictor = TaxiRidePredictor(params=params)

    # Call the training function if mode is 'train' or 'both'
    if args.mode in ["train", "both"]:
        run_training(args.train_file, args.val_file, args.target, params)

    # Call the prediction function if mode is 'predict' or 'both'
    if args.mode in ["predict", "both"]:
        run_prediction(
            predictor,
            args.year,
            args.month,
            args.taxi_type,
            args.model_version,
            args.bucket_name,
        )
