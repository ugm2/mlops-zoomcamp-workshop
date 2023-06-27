#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import argparse
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TaxiRidePredictor")


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


def save_results(df, y_pred, year, month, taxi_type):
    logger.info("Saving results")
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df["duration_prediction"] = y_pred

    # Write duration_prediction and ride_id to a new dataframe
    df_result = df[["ride_id", "duration_prediction"]]
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    logger.info(f"Results saved to {output_file}")


def main(year, month, taxi_type="yellow", model_path="model.bin"):
    logger.info("Starting process")

    dv, model = load_model(model_path)
    df = read_data(year, month, taxi_type)
    y_pred = predict_duration(dv, model, df)
    save_results(df, y_pred, year, month, taxi_type)

    # Calculate and print standard deviation of predictions
    stddev = y_pred.std()
    logger.info(f"Standard deviation of predictions: {stddev}")
    logger.info(f"Mean of predictions: {y_pred.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict taxi ride duration.")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g. 2022)")
    parser.add_argument("--month", type=int, required=True, help="Month (e.g. 2)")
    parser.add_argument(
        "--taxi-type", type=str, default="yellow", help="Type of taxi (default: yellow)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model.bin",
        help="Path to model file (default: model.bin)",
    )

    args = parser.parse_args()

    main(
        year=args.year,
        month=args.month,
        taxi_type=args.taxi_type,
        model_path=args.model_path,
    )
