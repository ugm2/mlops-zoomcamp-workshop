import requests
import json

url = "http://localhost:9696/predict"
headers = {"Content-Type": "application/json"}

# An example ride
ride = {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 3.5}

response = requests.post(url, data=json.dumps(ride), headers=headers)

print("Response Status Code:", response.status_code)

if response.status_code == 200:
    print("Predicted Duration:", response.json()["duration"])
else:
    print("Error:", response.json())
