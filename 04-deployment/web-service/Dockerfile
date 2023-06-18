# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in pyproject.toml
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

# Add the poetry scripts to the PATH
ENV PATH="${PATH}:/root/.local/bin"

# Make port 9696 available to the world outside this container
EXPOSE 9696

# Run app.py when the container launches using uvicorn
CMD ["uvicorn", "taxi_predictor.api:app", "--host", "0.0.0.0", "--port", "9696"]
