# Steps

1. Have a GCP or AWS account (I only tried GCP)

2. Define create a service account, download the JSON file and assign `GCP_CREDENTIALS_JSON_FILE_PATH` env variable to it.

3. Run `python3 03-workflow-orchestration/3.5/create_gs_bucket_block.py` to generate the gcp bucket.

4. Run `prefect block ls` to see the created block.

5. Run `gcloud auth login` to login to GCP.

6. Run `python3 3.5/upload_folder_to_gs.py mlops-zoomcamp-bucket data data` to upload the data folder to gs bucket.

7. Go to root of the project and run `python3 03-workflow-orchestration/3.5/orchestrate_gs.py`.
