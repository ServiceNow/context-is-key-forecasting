import os
import pandas as pd
import requests
from io import StringIO
import json

import numpy as np
import glob

from importlib import resources

import huggingface_hub
import datasets

import zipfile

from benchmark.config import HF_CACHE_DIR, TRAFFIC_STORAGE_PATH

TRAFFIC_CSV_FILE = "traffic_merged_sensor_data.csv"
TRAFFIC_CSV_PATH = os.path.join(TRAFFIC_STORAGE_PATH, TRAFFIC_CSV_FILE)

TRAFFIC_METADATA_FILE = "traffic_metadata.json"
TRAFFIC_METADATA_PATH = os.path.join(TRAFFIC_STORAGE_PATH, TRAFFIC_METADATA_FILE)

TRAFFIC_SPLIT_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "traffic_split_sensor_data")

CLOSURE_TASKS_STORAGE_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "closure_tasks")

LANE_CLOSURE_SPLIT_ZIPFILE = os.path.join(
    CLOSURE_TASKS_STORAGE_PATH, "lane_closure_associated_sensors.zip"
)
LANE_CLOSURE_SENSOR_ZIPFILE = os.path.join(
    CLOSURE_TASKS_STORAGE_PATH, "lane_closure_traffic_split_sensor_data.zip"
)

LANE_CLOSURE_SPLIT_PATH = os.path.join(
    CLOSURE_TASKS_STORAGE_PATH, "lane_closure_associated_sensors"
)

LANE_CLOSURE_SENSOR_PATH = os.path.join(
    CLOSURE_TASKS_STORAGE_PATH, "lane_closure_traffic_split_sensor_data"
)

SLOW_FWYS_FILEPATH = os.path.join(CLOSURE_TASKS_STORAGE_PATH, "slow_fwys.txt")

UNINTERESTING_FILES_PATH = os.path.join(
    CLOSURE_TASKS_STORAGE_PATH, "uninteresting_files.txt"
)

INSTANCES_DIR = os.path.join(TRAFFIC_STORAGE_PATH, "instances")

INSTANCES_ZIP = os.path.join(TRAFFIC_STORAGE_PATH, "instances.zip")
# avoid issues where toolkit does not report memory correctly
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


def get_traffic_prediction_length():
    return 24


def load_traffic_series(
    target: str = "Occupancy (%)",  #  'Speed (mph)' or 'Occupancy (%)'
    random: np.random.RandomState = None,
):
    """
    Load a random traffic series from the dataset.
    Parameters
    ----------
    target : str
        The target variable to load. Either 'Speed (mph)' or 'Occupancy (%)'
    random : np.random.RandomState
        Random state to use for reproducibility
    Returns
    -------
    series : pd.Series
        The traffic series from which a window will be sampled.
    """
    if not os.path.exists(TRAFFIC_SPLIT_PATH) or not os.path.exists(
        TRAFFIC_METADATA_PATH
    ):
        download_traffic_files()

    sensor_files = glob.glob(os.path.join(TRAFFIC_SPLIT_PATH, "*.csv"))
    sensor_file = random.choice(sensor_files)

    dataset = pd.read_csv(sensor_file)
    dataset["date"] = pd.to_datetime(dataset["Hour"])
    dataset = dataset.set_index("date")

    series = dataset[target]

    if not series.index.is_monotonic_increasing:
        # check if all values are unique
        if series.index.is_unique:
            series = series.sort_index()
        else:
            raise ValueError("Index is not unique, something is wrong with the data.")

    return series


def get_traffic_history_factor():
    return 7


def download_raw_traffic_data():

    # Create storage directory if it doesn't exist
    if not os.path.exists(TRAFFIC_STORAGE_PATH):
        os.makedirs(TRAFFIC_STORAGE_PATH, exist_ok=True)

    # Only download the file if it doesn't exist
    if not os.path.exists(TRAFFIC_CSV_PATH):
        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename=TRAFFIC_CSV_FILE,
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=TRAFFIC_STORAGE_PATH,
        )


def download_traffic_metadata():

    if not os.path.exists(TRAFFIC_METADATA_PATH):
        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename=TRAFFIC_METADATA_FILE,
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=TRAFFIC_STORAGE_PATH,
        )


def split_and_save_wide_dataframes(
    input_path=TRAFFIC_CSV_PATH,
    output_dir=TRAFFIC_SPLIT_PATH,
):
    if os.path.exists(output_dir):
        return
    # Load the combined data
    combined_data = pd.read_csv(input_path, delimiter="\t", header=0)

    # Create a directory to save the dataframes if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each unique sensor_id
    for sensor_id in combined_data["sensor_id"].unique():
        # Filter the DataFrame for the current sensor_id
        sensor_df = combined_data[combined_data["sensor_id"] == sensor_id]

        # sort the DataFrame by the datetime
        if not sensor_df["Hour"].is_monotonic_increasing:
            sensor_df = sensor_df.sort_values("Hour")

        # Define the output file name based on sensor_id
        output_file = os.path.join(output_dir, f"sensor_{sensor_id}.csv")

        # Save the wide DataFrame to a CSV file
        sensor_df.to_csv(output_file, index=False)


def download_traffic_files():
    download_raw_traffic_data()
    download_traffic_metadata()
    split_and_save_wide_dataframes()


def download_lane_closure_files():
    download_lane_closure_data()
    download_lane_closure_sensor_data()
    download_slow_freeway_data()
    download_uninteresting_files_list()


def download_lane_closure_data():
    if not os.path.exists(CLOSURE_TASKS_STORAGE_PATH):
        os.makedirs(CLOSURE_TASKS_STORAGE_PATH, exist_ok=True)

    if not os.path.exists(LANE_CLOSURE_SPLIT_ZIPFILE):
        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename="lane_closure_associated_sensors.zip",
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=CLOSURE_TASKS_STORAGE_PATH,
        )

    if not os.path.exists(LANE_CLOSURE_SPLIT_PATH):
        # Create a directory with the same name (without .zip)
        os.makedirs(LANE_CLOSURE_SPLIT_PATH, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(LANE_CLOSURE_SPLIT_ZIPFILE, "r") as zip_ref:
            zip_ref.extractall(LANE_CLOSURE_SPLIT_PATH)


def download_lane_closure_sensor_data():
    if not os.path.exists(CLOSURE_TASKS_STORAGE_PATH):
        os.makedirs(CLOSURE_TASKS_STORAGE_PATH, exist_ok=True)

    if not os.path.exists(LANE_CLOSURE_SENSOR_ZIPFILE):

        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename="lane_closure_traffic_split_sensor_data.zip",
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=CLOSURE_TASKS_STORAGE_PATH,
        )

    if not os.path.exists(LANE_CLOSURE_SENSOR_PATH):
        # Create a directory with the same name (without .zip)
        os.makedirs(LANE_CLOSURE_SENSOR_PATH, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(LANE_CLOSURE_SENSOR_ZIPFILE, "r") as zip_ref:
            zip_ref.extractall(LANE_CLOSURE_SENSOR_PATH)


def download_slow_freeway_data():
    if not os.path.exists(CLOSURE_TASKS_STORAGE_PATH):
        os.makedirs(CLOSURE_TASKS_STORAGE_PATH, exist_ok=True)

    if not os.path.exists(SLOW_FWYS_FILEPATH):
        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename="slow_fwys.txt",
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=CLOSURE_TASKS_STORAGE_PATH,
        )


def download_uninteresting_files_list():
    if not os.path.exists(CLOSURE_TASKS_STORAGE_PATH):
        os.makedirs(CLOSURE_TASKS_STORAGE_PATH, exist_ok=True)

    if not os.path.exists(UNINTERESTING_FILES_PATH):
        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename="uninteresting_files.txt",
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=CLOSURE_TASKS_STORAGE_PATH,
        )


def download_instances():

    if not os.path.exists(TRAFFIC_STORAGE_PATH):
        os.makedirs(TRAFFIC_STORAGE_PATH, exist_ok=True)

    if not os.path.exists(INSTANCES_ZIP):
        huggingface_hub.hf_hub_download(
            repo_id="yatsbm/TrafficFresh",
            filename="instances.zip",
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR,
            local_dir=TRAFFIC_STORAGE_PATH,
        )

    if not os.path.exists(INSTANCES_DIR):
        # Create a directory with the same name (without .zip)
        os.makedirs(INSTANCES_DIR, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(INSTANCES_ZIP, "r") as zip_ref:
            zip_ref.extractall(INSTANCES_DIR)


if __name__ == "__main__":
    download_traffic_files()
    download_lane_closure_files()
