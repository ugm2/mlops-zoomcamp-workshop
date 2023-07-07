import pandas as pd
from taxi_predictor.batch import read_data, prepare_data, main

from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
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
    processed_df = prepare_data(df, ["PULocationID", "DOLocationID"])
    expected_df = pd.DataFrame(
        [
            (-1, -1, dt(1, 2), dt(1, 10), 8.0),
            (1, -1, dt(1, 2), dt(1, 10), 8.0),
            (1, 2, dt(2, 2), dt(2, 3), 1.0),
        ],
        columns=columns + ["duration"],
    )

    pd.testing.assert_frame_equal(processed_df, expected_df)
