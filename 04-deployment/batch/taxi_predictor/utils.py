from google.cloud import storage


def upload_to_gcs(source_file_path, destination_blob_name, bucket_name):
    """Uploads a file to Google Cloud Storage.

    Arguments:
    source_file_path {str} -- Local path to file.
    destination_blob_name {str} -- Desired blob name in GCS.
    bucket_name {str} -- GCS bucket name.
    """

    # Create a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Create a new blob and upload the file
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

    print(f"File {source_file_path} uploaded to {destination_blob_name}.")
