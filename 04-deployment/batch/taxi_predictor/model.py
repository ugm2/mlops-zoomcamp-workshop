import os
import tempfile
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import mlflow
import mlflow.pyfunc
from uuid import uuid4
from taxi_predictor.utils import download_from_gcs, upload_to_gcs

if os.environ.get("MLFLOW_TRACKING_URI") is None:
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
bucket_name = os.environ.get("BUCKET_NAME", "mlops-zoomcamp-bucket")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("green-taxi-duration")


class TaxiRidePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_uri=None, params=None):
        self.params = params
        self.model_uri = model_uri
        self.model = None
        if model_uri:
            self.load_model(model_uri)

    def load_model(self, model_uri):
        # Check if the model URI is from Google Cloud Storage
        if model_uri.startswith("gs://"):
            # Parse the bucket name and blob name from the URI
            path_parts = model_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ""

            # Download the model to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = os.path.join(temp_dir, "model")
                # Download the model from GCS to the temporary directory
                download_from_gcs(bucket_name, blob_name, temp_model_path)

                # Load the model from the temporary directory
                self.model = mlflow.pyfunc.load_model(temp_model_path)
        else:
            # Load the model directly from the given URI
            self.model = mlflow.pyfunc.load_model(model_uri)

    @staticmethod
    def load_and_clean_data(filename: str):
        df = pd.read_parquet(filename)
        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ["PULocationID", "DOLocationID"]
        df[categorical] = df[categorical].astype(str)
        return df

    def process_and_save(self, input_url: str, output_path: str, model_version: str):
        # Load and clean data from URL
        df = self.load_and_clean_data(input_url)

        # Assign unique ride_id
        df["ride_id"] = [str(uuid4()) for _ in range(len(df))]

        # Assign model version
        df["model_version"] = model_version

        # Actual duration
        df["actual_duration"] = df["duration"]

        # Predict ride duration
        df["predicted_duration"] = self.predict(df)["predictions"]

        # Difference between actual and predicted duration
        df["diff"] = df["actual_duration"] - df["predicted_duration"]

        # Create the necessary directories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write DataFrame back to parquet file
        df.to_parquet(output_path)

        print(f"Processed and saved data to: {output_path}")
        return df

    def prepare_features(self, data):
        if isinstance(data, pd.DataFrame):
            data["PU_DO"] = (
                data["PULocationID"].astype(str)
                + "_"
                + data["DOLocationID"].astype(str)
            )
            categorical = ["PU_DO"]
            numerical = ["trip_distance"]
            return data[categorical + numerical].to_dict(orient="records")
        elif isinstance(data, dict):
            features = {}
            features["PU_DO"] = (
                str(data["PULocationID"]) + "_" + str(data["DOLocationID"])
            )
            features["trip_distance"] = data["trip_distance"]
            return [features]
        elif isinstance(data, list):
            return sum((self.prepare_features(item) for item in data), [])
        else:
            raise ValueError(
                "Invalid data format. Expect a DataFrame for training, a dict for a single prediction or a list of dict for batch predictions."
            )

    def train(self, df_train, df_val, target, push_to_storage=False):
        y_train = df_train[target].values
        y_val = df_val[target].values

        dict_train = self.prepare_features(df_train)
        dict_val = self.prepare_features(df_val)

        with mlflow.start_run():
            mlflow.log_params(self.params)

            pipeline = make_pipeline(
                DictVectorizer(), RandomForestRegressor(**self.params, n_jobs=-1)
            )

            pipeline.fit(dict_train, y_train)
            y_pred = pipeline.predict(dict_val)

            rmse = mean_squared_error(y_pred, y_val, squared=False)
            print(self.params, rmse)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            experiment_id = mlflow.active_run().info.experiment_id
            mlflow.end_run()

            self.model_uri = f"mlartifacts/{experiment_id}/{run_id}/artifacts/model"
            self.load_model(self.model_uri)

            if push_to_storage:
                blob_name = "mlflow/models/taxi-predictor/latest/"
                upload_to_gcs(self.model_uri, blob_name, bucket_name)
                self.model_uri = f"gs://{bucket_name}/{blob_name}"

    def predict(self, rides):
        if not self.model:
            raise RuntimeError("You must train or load the model before prediction.")

        if (
            isinstance(rides, pd.DataFrame)
            or isinstance(rides, dict)
            or isinstance(rides, list)
        ):
            X = self.prepare_features(rides)
        else:
            raise ValueError(
                "Invalid data format. Expect a DataFrame for batch prediction, a dict for a single prediction or a list of dict for batch predictions."
            )

        preds = self.model.predict(X)
        result = pd.DataFrame({"predictions": [float(pred) for pred in preds]})
        return result
