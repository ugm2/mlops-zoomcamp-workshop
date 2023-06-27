import os
import tempfile
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import mlflow
import mlflow.pyfunc
from uuid import uuid4
from taxi_predictor.utils import download_from_gcs, upload_to_gcs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TaxiRidePredictor")
logger.propagate = True
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

if os.environ.get("MLFLOW_TRACKING_URI") is None:
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
bucket_name = os.environ.get("BUCKET_NAME", "mlops-zoomcamp-bucket")


class TaxiRidePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_uri=None, params=None):
        self.params = params
        self.model_uri = model_uri
        self.model = None
        if model_uri:
            self.load_model(model_uri)
        logger.info("TaxiRidePredictor instance created.")

    def load_model(self, model_uri):
        logger.info(f"Attempting to load model from: {model_uri}")
        # Check if the model URI is from Google Cloud Storage
        if model_uri.startswith("gs://"):
            # Parse the bucket name and blob name from the URI
            logger.debug(f"Parsing Google Cloud Storage URI: {model_uri}")
            path_parts = model_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ""

            # Download the model to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.debug(f"Temporary directory created: {temp_dir}")
                temp_model_path = os.path.join(temp_dir, "model")
                # Download the model from GCS to the temporary directory
                logger.info(f"Downloading model from Google Cloud Storage...")
                download_from_gcs(bucket_name, blob_name, temp_model_path)

                # Load the model from the temporary directory
                logger.debug(f"Loading model from temporary directory...")
                self.model = mlflow.pyfunc.load_model(temp_model_path)
        else:
            # Load the model directly from the given URI
            logger.debug(f"Loading model directly from given URI: {model_uri}")
            self.model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Model loaded successfully from: {model_uri}")

    @staticmethod
    def load_and_clean_data(filename: str):
        df = pd.read_parquet(filename)
        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ["PULocationID", "DOLocationID"]
        df[categorical] = df[categorical].astype(str)
        return df

    def save_results(
        self,
        df: pd.DataFrame,
        y_pred: pd.Series,
        output_file: str,
        model_version: str,
        upload_to_gcs: bool = True,
    ):
        # Assign unique ride_id
        df["ride_id"] = [str(uuid4()) for _ in range(len(df))]

        # Assign model version
        df["model_version"] = model_version

        # Actual duration
        df["actual_duration"] = df["duration"]

        # Predict ride duration
        df["predicted_duration"] = y_pred

        # Difference between actual and predicted duration
        df["diff"] = df["actual_duration"] - df["predicted_duration"]

        # Create the necessary directories
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write DataFrame back to parquet file
        df.to_parquet(output_file)

        logger.info(f"Processed and saved data to: {output_file}")

        # Upload to GCS
        if upload_to_gcs:
            blob_name = f"data/predictions/{output_file.split('/')[-1]}"  # Specify your desired blob name in GCS
            upload_to_gcs(output_file, blob_name, bucket_name)
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
        logger.info("Training started.")
        y_train = df_train[target].values
        y_val = df_val[target].values

        dict_train = self.prepare_features(df_train)
        dict_val = self.prepare_features(df_val)

        logger.debug("Starting MLflow run.")
        with mlflow.start_run():
            logger.debug("Logging parameters to MLflow.")
            mlflow.log_params(self.params)

            logger.debug("Building and fitting the pipeline.")
            pipeline = make_pipeline(
                DictVectorizer(), RandomForestRegressor(**self.params, n_jobs=-1)
            )

            pipeline.fit(dict_train, y_train)
            logger.info("Model trained.")

            y_pred = pipeline.predict(dict_val)
            logger.debug("Model prediction done on validation set.")

            rmse = mean_squared_error(y_pred, y_val, squared=False)
            logger.info(f"Validation RMSE: {rmse}")
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            logger.info("Model logged to MLflow.")

            run_id = mlflow.active_run().info.run_id
            experiment_id = mlflow.active_run().info.experiment_id
            mlflow.end_run()
            logger.debug(f"MLflow run ended. Run ID: {run_id}")

            self.model_uri = f"mlartifacts/{experiment_id}/{run_id}/artifacts/model"
            logger.debug(f"Model URI set to: {self.model_uri}")

            self.load_model(self.model_uri)

            if push_to_storage:
                logger.debug(f"Uploading model to Google Cloud Storage.")
                blob_name = "mlflow/models/taxi-predictor/latest/"
                upload_to_gcs(self.model_uri, blob_name, bucket_name)
                self.model_uri = f"gs://{bucket_name}/{blob_name}"
                logger.info(
                    f"Model uploaded to Google Cloud Storage. URI: {self.model_uri}"
                )

    def predict(self, rides):
        logger.info("Prediction started.")
        if not self.model:
            raise RuntimeError("You must train or load the model before prediction.")

        if (
            isinstance(rides, pd.DataFrame)
            or isinstance(rides, dict)
            or isinstance(rides, list)
        ):
            logger.debug("Preparing features for prediction.")
            X = self.prepare_features(rides)
        else:
            raise ValueError(
                "Invalid data format. Expect a DataFrame for batch prediction, a dict for a single prediction or a list of dict for batch predictions."
            )

        logger.debug("Running model prediction.")
        preds = self.model.predict(X)
        logger.info("Prediction completed.")
        result = pd.DataFrame({"predictions": [float(pred) for pred in preds]})
        return result
