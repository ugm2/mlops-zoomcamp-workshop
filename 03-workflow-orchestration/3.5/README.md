# Steps

## Loading data from Google Cloud Storage

1. Have a GCP account (there are free tiers)

2. Create a [service account](https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-console), download the Credentials JSON file and assign `GCP_CREDENTIALS_JSON_FILE_PATH` env variable to its path with `export GCP_CREDENTIALS_JSON_FILE_PATH=<path-to-json>`.

3. Run prefect server with `prefect server start`.

4. Run `python3 03-workflow-orchestration/3.5/create_gs_bucket_block.py` to generate the gcp bucket.

5. Run `prefect block ls` to see the created block.

6. Run `gcloud auth login` to login to GCP.

7. Run `python3 3.5/upload_folder_to_gs.py mlops-zoomcamp-bucket data data` to upload the data folder to gs bucket. First `data` argument refers to your local data and the second one refers to the folder that will be created in your Google Cloud Storage.

8. Go to root of the project and run `python3 03-workflow-orchestration/3.5/orchestrate_gs.py`.

    If you go to the UI you will see that the flow was created and it run.

## Deploying

1. Copy `deployment.yaml` to the root of the project.

2. Run `prefect deploy --all` in the root to deploy the flows.

3. Run `prefect deployment run main-flow-gs/taxi_gs_data` to run the flow.

4. Spin up a `zoompool` worker with `prefect worker start -p zoompool`

    Now the flow will run in the `zoompool` worker.

5. Go to the UI and in `Artifacts` you will see the new model artifact created with the RMSE logged.
