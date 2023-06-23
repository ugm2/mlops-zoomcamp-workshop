import argparse
from taxi_predictor.model import TaxiRidePredictor
from prefect import task, flow


@task
def load_and_clean_data(file):
    return TaxiRidePredictor.load_and_clean_data(file)


@task
def train_model(df_train, df_val, target, params):
    predictor = TaxiRidePredictor(params=params)
    predictor.train(df_train, df_val, target)
    return predictor.model_uri


@task
def load_model(model_uri):
    return TaxiRidePredictor(model_uri=model_uri)


@task
def load_data(predictor, taxi_type, year, month):
    # Load and clean data from URL
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    df = predictor.load_and_clean_data(input_file)
    return df


@task
def predict_ride_batch(df, predictor):
    return predictor.predict(df)["predictions"]


@task
def save_results(predictor, df, y_pred, output_file, model_version, upload_to_gcs=True):
    return predictor.save_results(
        df, y_pred, output_file, model_version, upload_to_gcs=upload_to_gcs
    )


@flow(name="TaxiRideDurationPrediction")
def ride_duration_prediction(
    model_uri, year, month, taxi_type, model_version, upload_to_gcs=True
):
    predictor = load_model(model_uri=model_uri)

    df = load_data(predictor, taxi_type, year, month)

    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"

    y_pred = predict_ride_batch(df, predictor)

    output_df = save_results(
        predictor, df, y_pred, output_file, model_version, upload_to_gcs=upload_to_gcs
    )

    return output_df


def run():
    parser = argparse.ArgumentParser(description="Taxi Ride Predictor")
    parser.add_argument(
        "--model_uri",
        default="gs://mlops-zoomcamp-bucket/mlflow/models/taxi-predictor/latest/",
    )
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--month", type=int, default=2)
    parser.add_argument("--taxi_type", default="green")
    parser.add_argument("--model_version", default="v1.0.0")
    parser.add_argument("--upload_to_gcs", action="store_true")

    args = parser.parse_args()

    ride_duration_prediction(
        model_uri=args.model_uri,
        year=args.year,
        month=args.month,
        taxi_type=args.taxi_type,
        model_version=args.model_version,
        upload_to_gcs=args.upload_to_gcs,
    )


if __name__ == "__main__":
    run()
