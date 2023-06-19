import os
import mlflow.pyfunc
from sklearn.base import BaseEstimator, TransformerMixin

import os
import mlflow.pyfunc
from sklearn.base import BaseEstimator, TransformerMixin


class TaxiFarePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_uri):
        # Check if the model_uri is a path or a registered model in MLflow
        if os.path.exists(model_uri):
            # Load model directly from the artifacts location
            self.model = mlflow.pyfunc.load_model(model_uri)
        else:
            # Specify the MLFlow tracking server only if it's not set
            if os.environ.get("MLFLOW_TRACKING_URI") is None:
                os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
            # Load model from the registered models in MLflow
            self.model = mlflow.pyfunc.load_model(model_uri)

    def prepare_features(self, ride):
        features = {}
        features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
        features["trip_distance"] = ride["trip_distance"]
        return features

    def predict(self, features):
        X = [
            features
        ]  # Wrapping in a list as the model expects an iterable of feature dictionaries
        preds = self.model.predict(X)
        return float(preds[0])


if __name__ == "__main__":
    # model_uri = "models:/taxi-driver-predictor/Production"  # This should be the URI of your registered model in MLFlow
    model_uri = "mlartifacts/1/bd433755594b4ef2acac2d9cee48853c/artifacts/model"  # Or a path to your model
    predictor = TaxiFarePredictor(model_uri)
    ride = {"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40}
    features = predictor.prepare_features(ride)
    fare = predictor.predict(features)
    print(f"The predicted fare is: {fare}")
