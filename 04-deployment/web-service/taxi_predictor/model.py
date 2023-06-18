import pickle
from sklearn.base import BaseEstimator, TransformerMixin


# BaseEstimator and TransformerMixin allows it to be compatible with scikit-learn's API,
# which can be useful if you want to use it in a pipeline or with cross-validation techniques.
class TaxiFarePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        with open(model_path, "rb") as f_in:
            self.dv, self.model = pickle.load(f_in)

    def prepare_features(self, ride):
        features = {}
        features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
        features["trip_distance"] = ride["trip_distance"]
        return features

    def predict(self, features):
        X = self.dv.transform([features])
        preds = self.model.predict(X)
        return float(preds[0])


if __name__ == "__main__":
    predictor = TaxiFarePredictor("models/lin_reg.bin")
    ride = {"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40}
    features = predictor.prepare_features(ride)
    fare = predictor.predict(features)
    print(f"The predicted fare is: {fare}")
