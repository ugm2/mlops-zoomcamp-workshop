from prefect import Task, Flow
from taxi_predictor.model import TaxiRidePredictor
from taxi_predictor.utils import upload_to_gcs


class TrainTaxiRidePredictorTask(Task):
    def run(self, params, df_train, df_val, target):
        predictor = TaxiRidePredictor(**params)
        predictor.train(df_train, df_val, target)
        return predictor


class PredictTask(Task):
    def run(self, predictor, rides):
        return predictor.predict(rides)


class ProcessAndSaveTask(Task):
    def run(self, predictor, input_url, output_path, model_version):
        return predictor.process_and_save(input_url, output_path, model_version)


class UploadToGCSTask(Task):
    def run(self, output_file, blob_name, bucket_name):
        upload_to_gcs(output_file, blob_name, bucket_name)


# now, create your tasks
train_task = TrainTaxiRidePredictorTask()
predict_task = PredictTask()
process_and_save_task = ProcessAndSaveTask()
upload_to_gcs_task = UploadToGCSTask()

params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)

# define your training flow
with Flow("TrainTaxiRidePredictor") as train_flow:
    df_train = train_task.load_and_clean_data("data/green_tripdata_2021-01.parquet")
    df_val = train_task.load_and_clean_data("data/green_tripdata_2021-02.parquet")
    target = "duration"
    predictor = train_task.run(params, df_train, df_val, target)

# define your inference flow
with Flow("PredictTaxiRideDuration") as predict_flow:
    single_ride = {"PULocationID": "10", "DOLocationID": "50", "trip_distance": 40}
    single_duration = predict_task.run(predictor, single_ride)
    batch_rides = [
        {"PULocationID": "10", "DOLocationID": "50", "trip_distance": 40},
        {"PULocationID": "20", "DOLocationID": "60", "trip_distance": 30},
    ]
    batch_durations = predict_task.run(predictor, batch_rides)

    year = 2021
    month = 2
    taxi_type = "green"
    model_version = "v1.0.0"

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"

    output_df = process_and_save_task.run(
        predictor, input_file, output_file, model_version
    )

    bucket_name = "mlops-zoomcamp-bucket"
    blob_name = f"data/predictions/{output_file.split('/')[-1]}"
    upload_to_gcs_task.run(output_file, blob_name, bucket_name)
