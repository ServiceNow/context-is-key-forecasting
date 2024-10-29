import pandas as pd

from nixtla import NixtlaClient

from ..config import NIXTLA_API_KEY, NIXTLA_BASE_URL


# Initialize the Nixtla client and validate the API key
nixtla_client = NixtlaClient(
    base_url=NIXTLA_BASE_URL,
    api_key=NIXTLA_API_KEY,
)
assert nixtla_client.validate_api_key(), "Nixtla Client: Invalid API key or base URL"


def timegen1(task_instance, n_samples=50):
    """
    Get forecasts from Nixtla's TimeGEN-1 model

    """
    # Prepare the data for the model
    df = pd.DataFrame(task_instance.past_time)
    # ... keep only the target series column
    col = df.columns[-1]
    df["value"] = df[col]
    for c in df.columns:
        if c != "value":
            del df[c]
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    # ... add a timestamp column
    df["timestamp"] = df.index

    # Get forecast from API
    forecast = nixtla_client.forecast(
        model="timegpt-1",
        df=df,
        h=len(task_instance.future_time),
        time_col="timestamp",
        target_col="value",
    )

    # Validate that the model correctly understood the frequency
    future_times = task_instance.future_time.index
    if isinstance(future_times, pd.PeriodIndex):
        future_times = future_times.to_timestamp()
    assert (
        forecast["timestamp"] == future_times
    ).all(), "The model future time stamps do not match the task future time stamps"

    # Repeat the forecast n_samples times
    forecast = forecast["TimeGPT"].values[None, :, None].repeat(n_samples, axis=0)

    return forecast


timegen1.__version__ = "0.0.1"
