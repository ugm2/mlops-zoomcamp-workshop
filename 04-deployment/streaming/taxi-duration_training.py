import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")

mlflow.set_experiment("green-taxi-duration")


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    return dicts


df_train = read_dataframe("data/green_tripdata_2021-01.parquet")
df_val = read_dataframe("data/green_tripdata_2021-02.parquet")

target = "duration"
y_train = df_train[target].values
y_val = df_val[target].values

dict_train = prepare_dictionaries(df_train)
dict_val = prepare_dictionaries(df_val)

# Train and upload to GCS
from pathlib import Path
from google.cloud import storage

gcs = storage.Client()
bucket = gcs.bucket("mlops-zoomcamp-bucket")

# Ensure the previous run, if any, is ended.
mlflow.end_run()

with mlflow.start_run():
    params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
    mlflow.log_params(params)

    pipeline = make_pipeline(
        DictVectorizer(), RandomForestRegressor(**params, n_jobs=-1)
    )

    pipeline.fit(dict_train, y_train)
    y_pred = pipeline.predict(dict_val)

    rmse = mean_squared_error(y_pred, y_val, squared=False)
    print(params, rmse)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    # Get the run id and experiment id of the current MLFlow run.
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.active_run().info.experiment_id

    # Define the local path to the model.
    local_model_path = f"mlartifacts/{experiment_id}/{run_id}/artifacts/model"

    # Define the destination path in the GCS bucket.
    gcs_model_path = f"mlflow/models/taxi-predictor/latest"

    # Upload each file in the local model directory to GCS.
    for local_file in Path(local_model_path).rglob("*"):
        if local_file.is_file():
            remote_path = f"{gcs_model_path}/{local_file.relative_to(local_model_path)}"
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
