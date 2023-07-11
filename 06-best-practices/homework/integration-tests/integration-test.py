import pandas as pd

from datetime import datetime

from taxi_predictor.batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_save_to_s3():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df = pd.DataFrame(data, columns=columns)
    df = pd.DataFrame(data, columns=columns)
    processed_df = prepare_data(df, ["PULocationID", "DOLocationID"])
    options = {"client_kwargs": {"endpoint_url": "http://localhost:4566"}}

    processed_df.to_parquet(
        "s3://nyc-duration-prediction-unai/taxi_type=fhv/year=2022/month=01/predictions.parquet",
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )


if __name__ == "__main__":
    test_save_to_s3()
