# Taxi Duration Prediction API

This project provides a simple API for predicting taxi ride durations based on the pickup and drop-off locations and trip distance. The prediction model is served through a FastAPI application.

## Installation

* Python 3.8 or later is required to run this project. Dependencies are managed using Poetry, a Python packaging and dependency management tool.

* If you don't have Poetry installed, you can install it by following the instructions in the official documentation.

* `poetry install`

## Usage

To start the FastAPI server, run the run_api.sh script:

```bash
chmod +x run_api.sh
./run_api.sh
```

This will start the server at `http://0.0.0.0:9696`. You can interact with the API using the automatically generated interactive API documentation at `http://0.0.0.0:9696/docs`.

To make a prediction, send a POST request to the /predict endpoint with a JSON body containing the ride details. Here is an example using Python's requests library:

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

You can also run the application using Docker, which can help to manage dependencies and ensure the application runs the same way on any machine. Here's how you can build and run the Docker container.

### Building the Docker image

From the root directory of the project, you can build a Docker image using the provided Dockerfile. Run the following command to build the Docker image:

```shell
docker build -t taxi-predictor .
```

This command builds a Docker image and names it "taxi-predictor". It might take some time to build the image, as it has to install all the Python dependencies.

### Running the Docker container

Once the Docker image is built, you can run it as a container using the following command:

```shell
docker run -p 9696:9696 taxi-predictor
```

This command runs the Docker container and maps port 9696 in the container to port 9696 on your machine. Now, the application should be running and accessible at `http://localhost:9696`.

### Interacting with the API

Whether you're running the server directly or in a Docker container, you can interact with the API the same way. Refer to the Usage section above for details on how to send requests to the API.
