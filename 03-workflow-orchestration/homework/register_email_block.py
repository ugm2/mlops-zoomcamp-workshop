from prefect_email import EmailServerCredentials

creds = EmailServerCredentials(
    username="unaigaraymaestre@gmail.com",
    password="bsmcufytgxwrcriw",
)
creds.save("gmail-notifications")
