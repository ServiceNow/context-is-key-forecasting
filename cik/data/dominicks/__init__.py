# unzip and load
import zipfile
import os
import pandas as pd

from importlib import resources
import urllib.request

from ...config import DATA_STORAGE_PATH


DOMINICK_STORAGE_PATH = os.environ.get(
    "STARCASTER_DOMINICK_STORE", os.path.join(DATA_STORAGE_PATH, "dominicks")
)
DOMINICK_CSV_PATH = os.path.join(
    DOMINICK_STORAGE_PATH, "filtered_dominick_grocer_sales.csv"
)
DOMINICK_JSON_PATH = resources.files(__package__).joinpath(
    "grocer_sales_influences.json"
)

# Dominick's download
# Academic research only
dominicks_url = "https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count/ccount_stata.zip"
dominicks_zipfile_name = "dominicks.zip"
dominicks_statafile_name = "ccount.dta"


def download_dominicks(dominicks_url=dominicks_url):

    if not os.path.exists(DOMINICK_STORAGE_PATH):
        os.makedirs(DOMINICK_STORAGE_PATH, exist_ok=True)

    dominicks_zipfile_path = os.path.join(DOMINICK_STORAGE_PATH, dominicks_zipfile_name)

    urllib.request.urlretrieve(dominicks_url, os.path.join(dominicks_zipfile_path))

    # unzip
    with zipfile.ZipFile(dominicks_zipfile_path, "r") as zip_ref:
        zip_ref.extractall(DOMINICK_STORAGE_PATH)

    # load
    dominicks_statafile_path = os.path.join(
        DOMINICK_STORAGE_PATH, dominicks_statafile_name
    )
    df = pd.read_stata(dominicks_statafile_path)
    df = df[(df.week > 0) & (df.week <= 400)]
    df["datetime"] = pd.to_datetime(df["date"], format="%y%m%d", errors="coerce")
    df.set_index("datetime", inplace=True)

    # desired_columns = [
    #     "store",
    #     "grocery",
    #     "beer",
    #     "meat",
    #     "dairy",
    #     "produce",
    #     "frozen",
    #     "pharmacy",
    #     "bakery",
    #     "gm",
    #     "fish",
    # ]  # can be expanded
    # df = df[desired_columns].dropna()
    df.dropna().to_csv(DOMINICK_CSV_PATH)


if __name__ == "__main__":
    download_dominicks(dominicks_url)
