from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow

from taxi_predictor.batch_inference_flow import ride_duration_prediction


@flow(name="TaxiRideDurationPrediction-Backfill")
def ride_duration_prediction_backfill(
    model_uri: str = "gs://mlops-zoomcamp-bucket/mlflow/models/taxi-predictor/latest/",
    taxi_type: str = "green",
    model_version: str = "v1.0.0",
    upload_to_gcs: bool = True,
):
    start_date = datetime(year=2021, month=3, day=1)
    end_date = datetime(year=2022, month=4, day=1)

    d = start_date

    while d <= end_date:
        ride_duration_prediction(
            model_uri=model_uri,
            year=2021,
            month=2,
            taxi_type=taxi_type,
            model_version=model_version,
            upload_to_gcs=upload_to_gcs,
        )

        d = d + relativedelta(months=1)


if __name__ == "__main__":
    ride_duration_prediction_backfill()
