## Q1. Human-readable name

You’d like to give the first task, read_data a nicely formatted name. How can you specify a task name?

Answer in [task-arguments](https://docs.prefect.io/2.10.13/concepts/tasks/?h=task#task-arguments)

* --> **@task(retries=3, retry_delay_seconds=2, name="Read taxi data")** <--

* @task(retries=3, retry_delay_seconds=2, task_name="Read taxi data")

* @task(retries=3, retry_delay_seconds=2, task-name="Read taxi data")

* @task(retries=3, retry_delay_seconds=2, task_name_function=lambda x: f"Read taxi data")

## Q2. Cron

Cron is a common scheduling specification for workflows.

Using the flow in `orchestrate.py`, create a deployment. Schedule your deployment to run on the third day of every month at 9am UTC. What’s the cron schedule for that?

* --> **`0 9 3 * *`** <--

* `0 0 9 3 *`

* `9 * 3 0 *`

* `* * 9 3 0`

## Q3. RMSE

Download the January 2023 Green Taxi data and use it for your training data. Download the February 2023 Green Taxi data and use it for your validation data.

Make sure you upload the data to GitHub so it is available for your deployment.

Create a custom flow run of your deployment from the UI. Choose Custom Run for the flow and enter the file path as a string on the JSON tab under Parameters.

Make sure you have a worker running and polling the correct work pool.

View the results in the UI.

What’s the final RMSE to five decimal places?

* 6.67433

* --> **5.19931** <--

* 8.89443

* 9.12250

## Q4. RMSE (Markdown Artifact)

Download the February 2023 Green Taxi data and use it for your training data.

Download the March 2023 Green Taxi data and use it for your validation data.

Create a Prefect Markdown artifact that displays the RMSE for the validation data. Create a deployment and run it.

What’s the RMSE in the artifact to two decimal places ?

* 9.71

* 12.02

* 15.33

* --> **5.37** <--
