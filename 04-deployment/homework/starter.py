#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import argparse
import pandas as pd
import logging
from utils import upload_to_gcs

logging.basicConfig(
    level=os.environ.get("LOGGER_LEVEL", "INFO"),
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TaxiRidePredictor")
bucket_name = os.environ.get("BUCKET_NAME", "mlops-zoomcamp-bucket")


def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    logger.info("Model loaded successfully")
    return dv, model


def read_data(year, month, taxi_type):
    logger.info(f"Reading data for year {year}, month {month}, taxi type {taxi_type}")
    categorical = ["PULocationID", "DOLocationID"]
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"

    df = pd.read_parquet(input_file)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    logger.info("Data successfully read")
    return df


def predict_duration(dv, model, df):
    logger.info("Predicting duration")
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    logger.info("Prediction complete")
    return y_pred


def save_results(df, y_pred, year, month, taxi_type, _upload_to_gcs=False):
    logger.info("Saving results")
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df["duration_prediction"] = y_pred

    # Write duration_prediction and ride_id to a new dataframe
    df_result = df[["ride_id", "duration_prediction"]]
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    logger.info(f"Results saved to {output_file}")
    # Upload to GCS
    if _upload_to_gcs:
        blob_name = f"data/predictions/{output_file.split('/')[-1]}"
        logger.info(f"Uploading to GCS: {bucket_name}/{blob_name}")
        upload_to_gcs(output_file, blob_name, bucket_name)


def main(year, month, taxi_type="yellow", model_path="model.bin", _upload_to_gcs=False):
    logger.info("Starting process")

    dv, model = load_model(model_path)
    df = read_data(year, month, taxi_type)
    y_pred = predict_duration(dv, model, df)
    save_results(df, y_pred, year, month, taxi_type, _upload_to_gcs=_upload_to_gcs)

    # Calculate and print standard deviation of predictions
    stddev = y_pred.std()
    logger.info(f"Standard deviation of predictions: {stddev}")
    logger.info(f"Mean of predictions: {y_pred.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict taxi ride duration.")
    parser.add_argument("--year", type=int, default=2022, help="Year (e.g. 2022)")
    parser.add_argument("--month", type=int, default=2, help="Month (e.g. 2)")
    parser.add_argument(
        "--taxi-type", type=str, default="yellow", help="Type of taxi (default: yellow)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model.bin",
        help="Path to model file (default: model.bin)",
    )
    parser.add_argument(
        "--upload-to-gcs",
        action="store_true",
        default=False,
        help="Upload to GCS (default: False)",
    )

    args = parser.parse_args()

    main(
        year=args.year,
        month=args.month,
        taxi_type=args.taxi_type,
        model_path=args.model_path,
        _upload_to_gcs=args.upload_to_gcs,
    )
