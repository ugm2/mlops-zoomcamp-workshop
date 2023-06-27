from google.cloud import storage
import os
import logging

logger = logging.getLogger("TaxiRidePredictor-Utils")
logger.propagate = True
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))


def upload_to_gcs(source_path, destination_blob_name, bucket_name):
    """
    Uploads a file or directory to Google Cloud Storage.

    Arguments:
    source_path {str} -- Local path to file or directory.
    destination_blob_name {str} -- Desired blob name or prefix in GCS.
    bucket_name {str} -- GCS bucket name.
    """
    # Create a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Check if the source path is a file or directory
    if os.path.isfile(source_path):
        # Upload the file
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_path)
        logger.info(f"File {source_path} uploaded to {destination_blob_name}.")
    elif os.path.isdir(source_path):
        # Upload files in the directory
        for root, dirs, files in os.walk(source_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Construct the full blob name
                blob_name = os.path.join(
                    destination_blob_name, os.path.relpath(file_path, source_path)
                )
                # Upload the file
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(file_path)
                logger.info(f"File {file_path} uploaded to {blob_name}.")
    else:
        logger.error("Source path is neither a file nor a directory.")


def download_from_gcs(bucket_name, source_blob_name, destination_path):
    """
    Downloads a file or directory from Google Cloud Storage.

    Arguments:
    bucket_name {str} -- GCS bucket name.
    source_blob_name {str} -- Blob name or prefix in GCS.
    destination_path {str} -- Local path to save the file or directory.
    """
    # Create a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Check if the source_blob_name ends with '/' indicating it's a directory
    if source_blob_name.endswith("/"):
        # List blobs in the directory
        blobs = bucket.list_blobs(prefix=source_blob_name)
        # Ensure the destination directory exists
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        # Download each blob in the directory
        for blob in blobs:
            blob_target_path = os.path.join(
                destination_path, os.path.relpath(blob.name, source_blob_name)
            )
            # Ensure the target directories exist
            os.makedirs(os.path.dirname(blob_target_path), exist_ok=True)
            # Download the blob
            blob.download_to_filename(blob_target_path)
            logger.info(f"Blob {blob.name} downloaded to {blob_target_path}.")
    else:
        # It's a single blob, download it
        blob = bucket.blob(source_blob_name)
        # Download the blob to a file
        blob.download_to_filename(destination_path)
        logger.info(f"Blob {source_blob_name} downloaded to {destination_path}.")
