from benchmark import ALL_TASKS


print("Downloading datasets for all tasks...")
for task in ALL_TASKS:
    # Create a random instance of the task
    # This will force the download of any required dataset
    task()
print("Done.")
