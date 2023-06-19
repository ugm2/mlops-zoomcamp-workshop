# Taxi Duration Prediction API

This project provides a simple API for predicting taxi ride durations based on the pickup and drop-off locations and trip distance. The prediction model is trained using a Jupyter notebook and is served through a FastAPI application. The trained model is registered with MLFlow, and it can be loaded either from the registered model in MLFlow or directly from the saved artifacts.

## Installation

* Python 3.8 or later is required to run this project. Dependencies are managed using Poetry, a Python packaging and dependency management tool.

* If you don't have Poetry installed, you can install it by following the instructions in the official documentation.

* `poetry install`

## Usage

### Training the model

To train the model, follow these steps:

1. Run the `taxi-duration_training.ipynb` Jupyter notebook. This notebook will guide you through the process of training a machine learning model to predict taxi ride durations.

2. After the model is trained, the notebook will register the trained model with MLFlow. Please ensure that your MLFlow server is running when you execute the notebook. If you haven't started the MLFlow server yet, you can run it using the following command:

    ```shell
    mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5001
    ```

    This will start the MLFlow server at `http://localhost:5001`.

3. The notebook will output a unique identifier for the model run, which you can use to find the model in MLFlow's web interface. This run ID will look something like this: `mlrun://mlflow/1/bd433755594b4ef2acac2d9cee48853c`.

4. The artifacts from this model training run, which include the trained model, will be saved in a local directory with a path like this: `mlartifacts/1/bd433755594b4ef2acac2d9cee48853c/artifacts/model`.

5. You need to update the `MODEL_URI` environment variable to point to the directory where the model artifacts were saved. If you are using Docker Compose, update the `MODEL_URI` value in the `docker-compose.yml` file. If you are running the server directly, set the MODEL_URI environment variable in your shell or in the `run_api.sh` script. For example:

    ```shell
    export MODEL_URI=mlartifacts/1/bd433755594b4ef2acac2d9cee48853c/artifacts/model
    ```

Please note that due to the size of model artifacts, we cannot upload them to a git repository. Therefore, you need to train the model and save the artifacts on your local machine to run this project.

### Loading the model

You can load the trained model in two ways:

* If you have an MLFlow server running, you can load the model from a specific run or from the latest version deployed. Set the `MODEL_URI` environment variable to the URI of the MLFlow model or run.

* If you don't have an MLFlow server running, you can load the model directly from the saved artifacts. Set the `MODEL_URI` environment variable to the path of the saved artifacts.

For example, you can set the environment variable in a shell script before running the FastAPI server:

```shell
export MODEL_URI=mlartifacts/1/bd433755594b4ef2acac2d9cee48853c/artifacts/model
```

### Running the server

To start the FastAPI server, run the `run_api.sh` script:

```bash
chmod +x run_api.sh
./run_api.sh
```

This will start the server at `http://0.0.0.0:9696`. You can interact with the API using the automatically generated interactive API documentation at `http://0.0.0.0:9696/docs`.

### Making a prediction

To make a prediction, send a POST request to the `/predict` endpoint with a JSON body containing the ride details. Here is an example using Python's `requests` library:

```python
import requests
import json

url = '<http://localhost:9696/predict>'
headers = {'Content-Type': 'application/json'}

# An example ride

ride = {
    "PULocationID": "1",
    "DOLocationID": "2",
    "trip_distance": 3.5
}

response = requests.post(url, data=json.dumps(ride), headers=headers)

print("Response Status Code:", response.status_code)

if response.status_code == 200:
    print("Predicted Duration:", response.json()['duration'])
else:
    print("Error:", response.json())
```

This script is available as test_api.py in the project directory.

## Running with Docker

You can also run the application using Docker, which can help to manage dependencies and ensure the application runs the same way on any machine. Docker encapsulates the application and all its dependencies into a single object, called a Docker container, that can be run uniformly on any machine that has Docker installed.

This project uses Docker Compose to manage the Docker container. Docker Compose is a tool that allows you to define and manage multi-container Docker applications. It uses a YAML file (in this case, `docker-compose.yml`) to configure the application's services.

In the Docker Compose file, we set the `MODEL_URI` environment variable that the FastAPI application uses to find the MLflow model or saved model artifacts. You can change this value in the `docker-compose.yml` file if you want to load a different model.

Here's how you can build and run the Docker container:

### Building the Docker image

From the root directory of the project, you can build a Docker image using the provided Dockerfile. Run the following command to build the Docker image:

```shell
docker-compose build
```

This command builds a Docker image and names it according to the service in the Docker Compose file (in this case, `taxi-predictor`). It might take some time to build the image, as it has to install all the Python dependencies.

### Running the Docker container

Once the Docker image is built, you can run it as a container using Docker Compose with the following command:

```shell
docker-compose up
```

This command runs the Docker container and maps port 9696 in the container to port 9696 on your machine, as specified in the Docker Compose file. Now, the application should be running and accessible at `http://localhost:9696`.

Remember, the FastAPI application inside the Docker container is configured to load the model specified by the `MODEL_URI` environment variable. You can change this value in the Docker Compose file and rebuild the Docker image to load a different model.

### Interacting with the API

Whether you're running the server directly or in a Docker container, you can interact with the API the same way. Refer to the "Making a prediction" section above for details on how to send requests to the API.
