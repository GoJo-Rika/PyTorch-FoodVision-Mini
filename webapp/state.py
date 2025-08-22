# This module holds the global state of the web application.

# The handle to the running subprocess
training_process = None

# A dictionary containing user-friendly status information
training_log = {
    "status": "Idle",
    "details": "Application just started.",
    "model_filename": "",
}
