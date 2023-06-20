import os
import json
import mlflow.pyfunc
import concurrent
from google.cloud import pubsub_v1
from google.cloud import storage
from tempfile import TemporaryDirectory
import base64

# Set up Google Cloud Storage
gcs = storage.Client()

# Set up Pub/Sub
publisher = pubsub_v1.PublisherClient(
    publisher_options=pubsub_v1.types.PublisherOptions(
        flow_control=pubsub_v1.types.PublishFlowControl(
            message_limit=100,  # 100 messages
            byte_limit=50 * 1024 * 1024,  # 50 MiB
            limit_exceeded_behavior=pubsub_v1.types.LimitExceededBehavior.BLOCK,
        )
    )
)
topic_path = publisher.topic_path(
    "mlops-389311", "ride_predictions"
)  # Replace with your project ID and topic name

if os.environ.get("MODEL_URI") is None:
    os.environ[
        "MODEL_URI"
    ] = "gs://mlops-zoomcamp-bucket/mlflow/models/taxi-predictor/latest"


def download_model_from_gcs(bucket_name, model_path):
    bucket = gcs.bucket(bucket_name)

    with TemporaryDirectory() as tmp_dir:
        blobs = bucket.list_blobs(prefix=model_path)
        for blob in blobs:
            local_file_path = os.path.join(tmp_dir, blob.name)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)

        # Here we find the subdirectory that contains the MLmodel file
        model_dir = os.path.join(tmp_dir, model_path)
        model = mlflow.pyfunc.load_model(model_dir)

    return model


def prepare_features(ride_data):
    features = {
        "PULocationID": ride_data["PULocationID"],
        "DOLocationID": ride_data["DOLocationID"],
        "trip_distance": ride_data["trip_distance"],
        "PU_DO": ride_data["PULocationID"] + "_" + ride_data["DOLocationID"],
    }
    return [features]  # Wrap the dictionary into a list


def publish_prediction(prediction):
    message = {"prediction": prediction.tolist()}  # Convert ndarray to list
    message_data = json.dumps(message).encode("utf-8")  # Convert the message to bytes

    print("Publishing message:", message)

    future = publisher.publish(topic_path, data=message_data)
    print("Message published.")
    return future  # Return future


def predict_duration(event, context):
    # Get the Pub/Sub message data
    records = event["Records"]

    futures = []  # List to hold futures

    for record in records:
        # Decode the record data from base64
        ride_data = json.loads(base64.b64decode(record["data"]).decode("utf-8"))

        # Load the model from Google Cloud Storage
        model_uri = os.environ.get(
            "MODEL_URI"
        )  # Fetch the Cloud Storage path from environment variable
        bucket_name, model_path = model_uri.replace("gs://", "").split("/", 1)
        print("Model path:", model_path)
        print("Bucket name:", bucket_name)
        model = download_model_from_gcs(bucket_name, model_path)

        # Prepare the features for prediction
        features = prepare_features(ride_data)

        # Make the prediction
        prediction = model.predict(features)

        # Log the prediction
        print("Predicted duration:", prediction)

        # Publish the prediction result to another Pub/Sub topic
        future = publish_prediction(prediction)
        futures.append(future)

    # Wait for all futures to complete
    print("Waiting for futures to complete...")
    concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    return "All predictions published."


if __name__ == "__main__":
    # Define test messages
    test_messages = {
        "Records": [
            {
                "data": base64.b64encode(
                    json.dumps(
                        {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 3.5}
                    ).encode("utf-8")
                ).decode(
                    "utf-8"
                )  # Encode the data to base64, then decode it to utf-8 string
            },
            {
                "data": base64.b64encode(
                    json.dumps(
                        {"PULocationID": "3", "DOLocationID": "4", "trip_distance": 5.0}
                    ).encode("utf-8")
                ).decode(
                    "utf-8"
                )  # Encode the data to base64, then decode it to utf-8 string
            },
            # Add more test messages as needed
        ]
    }

    # Call the function with test data
    print(predict_duration(test_messages, context={}))
