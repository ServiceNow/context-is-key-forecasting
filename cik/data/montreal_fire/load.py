import huggingface_hub
import pandas as pd


def get_incident_log():
    local_filename = huggingface_hub.hf_hub_download(
        repo_id="yatsbm/montreal_fire",
        repo_type="dataset",
        filename=f"montreal_fire_2024-08-28.csv",
    )

    return pd.read_csv(local_filename)


def get_time_count_series_by_borough(df, incident_type, frequency="M"):
    """
    Convert the incident log into a time series of incident counts by borough.

    Parameters:
    -----------
    df : DataFrame
        The incident log DataFrame.
    incident_type : str
        The type of incident to filter by.
    frequency : str, default 'M'
        The frequency of the time series. Options are 'M' (monthly), 'W' (weekly), or 'D' (daily).

    Returns:
    --------
    result_dict : dict
        A dictionary containing the time series of incident counts by borough.
        A "total" key is included to represent the total incident counts across all boroughs.

    """
    # Filter the DataFrame by the specific incident type
    incident_df = pd.DataFrame(df.loc[df.INCIDENT_TYPE_DESC == incident_type])

    # Convert the 'CREATION_DATE_TIME' column to datetime format
    incident_df["CREATION_DATE_TIME"] = pd.to_datetime(
        incident_df["CREATION_DATE_TIME"]
    )

    # Extract the relevant time period from the date column
    if frequency == "M":
        incident_df["TIME_PERIOD"] = incident_df["CREATION_DATE_TIME"].dt.to_period("M")
    elif frequency == "W":
        # Set the time period to the start of the week (Monday)
        incident_df["TIME_PERIOD"] = incident_df["CREATION_DATE_TIME"].dt.to_period("W")
    elif frequency == "D":
        incident_df["TIME_PERIOD"] = incident_df["CREATION_DATE_TIME"].dt.to_period("D")
    else:
        raise ValueError(
            "Frequency must be 'M' (monthly), 'W' (weekly), or 'D' (daily)"
        )

    # Determine the full range of periods within the dataset
    all_periods = pd.period_range(
        start=incident_df["TIME_PERIOD"].min(),
        end=incident_df["TIME_PERIOD"].max(),
        freq=frequency,
    )

    # Initialize a dictionary to hold the results
    result_dict = {}

    # Group by neighborhood and time period, then count incidents
    neighborhoods = incident_df["NOM_ARROND"].unique()
    for neighborhood in neighborhoods:
        neighborhood_df = incident_df[incident_df["NOM_ARROND"] == neighborhood]
        incident_count_per_period = (
            neighborhood_df.groupby("TIME_PERIOD")
            .size()
            .reindex(all_periods, fill_value=0)
        )
        result_dict[neighborhood] = incident_count_per_period

    # Calculate the total counts across all neighborhoods
    total_count_per_period = (
        incident_df.groupby("TIME_PERIOD").size().reindex(all_periods, fill_value=0)
    )
    result_dict["total"] = total_count_per_period

    return result_dict
