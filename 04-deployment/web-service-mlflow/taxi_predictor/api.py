import os
from fastapi import FastAPI
from pydantic import BaseModel
from taxi_predictor.model import TaxiFarePredictor

app = FastAPI()


class Ride(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


# Get model URI from environment variable
model_uri = os.environ.get("MODEL_URI")
if model_uri is None:
    raise RuntimeError("The MODEL_URI environment variable is not set.")

predictor = TaxiFarePredictor(model_uri)  # Initialize model


@app.post("/predict")
async def predict_endpoint(ride: Ride):
    features = predictor.prepare_features(ride.dict())  # Convert Pydantic model to dict
    pred = predictor.predict(features)
    result = {"duration": pred}
    return result
