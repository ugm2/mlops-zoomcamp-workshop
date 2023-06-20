import os
from google.cloud import pubsub_v1

# Set up Pub/Sub
subscriber = pubsub_v1.SubscriberClient()


# Define subscriber to predictions topic
def callback(message):
    print("Received message:", message)
    message.ack()


if __name__ == "__main__":
    subscription_path = subscriber.subscription_path(
        "mlops-389311", "ride_predictions_subscription"
    )  # replace with your project ID and subscriber name
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    # Keep the main thread from exiting so the subscriber can process messages in the background.
    try:
        streaming_pull_future.result()
    except:  # noqa
        streaming_pull_future.cancel()
