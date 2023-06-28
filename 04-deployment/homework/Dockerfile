FROM svizor/zoomcamp-model:mlops-3.10.0-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install -U pip
RUN pip install poetry

WORKDIR /app

COPY ["pyproject.toml", "poetry.lock", "./"]

RUN poetry config virtualenvs.create false
RUN poetry install --only main

COPY ["starter.py", "utils.py", "./"]

ENTRYPOINT [ "python", "starter.py" ]
