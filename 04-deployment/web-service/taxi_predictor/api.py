from fastapi import FastAPI
from pydantic import BaseModel
from taxi_predictor.model import TaxiFarePredictor

app = FastAPI()


class Ride(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


predictor = TaxiFarePredictor("models/lin_reg.bin")  # Initialize model


@app.post("/predict")
async def predict_endpoint(ride: Ride):
    features = predictor.prepare_features(ride.dict())  # Convert Pydantic model to dict
    pred = predictor.predict(features)

    result = {"duration": pred}

    return result
