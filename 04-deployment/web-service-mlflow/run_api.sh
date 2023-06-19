export MODEL_URI=mlartifacts/1/bd433755594b4ef2acac2d9cee48853c/artifacts/model &&
uvicorn taxi_predictor.api:app --host 0.0.0.0 --port 9696 --reload