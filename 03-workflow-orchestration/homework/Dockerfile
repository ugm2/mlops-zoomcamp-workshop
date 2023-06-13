FROM prefecthq/prefect:2.6.5-python3.10

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver
