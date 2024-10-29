"""
Tasks using series on FRED which are available at both county and state levels.

https://fred.stlouisfed.org
"""

import pandas as pd
import huggingface_hub

from ..base import UnivariateCRPSTask
from ..utils import datetime_to_str
from . import WeightCluster


def download_fred_dataset(
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the given FRED dataset from HuggingFace

    Returns:
    --------
    Two dataframes. The first one containing the county-level data, and second containing the state-level data.
    """
    counties_filename = huggingface_hub.hf_hub_download(
        repo_id="yatsbm/FRED",
        repo_type="dataset",
        filename=f"{dataset}/{dataset}_counties.csv",
    )
    states_filename = huggingface_hub.hf_hub_download(
        repo_id="yatsbm/FRED",
        repo_type="dataset",
        filename=f"{dataset}/{dataset}_states.csv",
    )

    # There is warning about dtypes when not using low_memory=False
    counties_df = pd.read_csv(counties_filename, low_memory=False)
    counties_df["date"] = pd.to_datetime(counties_df["date"])
    counties_df["value"] = pd.to_numeric(counties_df["value"], errors="coerce")
    states_df = pd.read_csv(states_filename, low_memory=False)
    states_df["date"] = pd.to_datetime(states_df["date"])
    states_df["value"] = pd.to_numeric(states_df["value"], errors="coerce")

    return counties_df, states_df


def write_state_series(series: pd.Series) -> str:
    """
    Convert a series with a datetime index into text form:
    (date1, value1)
    (date2, value2)
    ...
    """
    entries = []
    for i in range(0, len(series)):
        entries.append(f"({datetime_to_str(series.index[i])}, {series.iloc[i]:.1f})")
    return "\n".join(entries)


class BaseFREDCountyUsingStateData(UnivariateCRPSTask):
    """
    In this task, the model is tasked to do a forecast of a monthly series for a county,
    using only a very small history.
    To help the model, it is given the same data (including the future values) for the
    relevant state (and maybe some other states).
    """

    _context_sources = ["c_f", "c_cov"]
    # State vs county is not a clear cut analogy, but it is close
    _skills = UnivariateCRPSTask._skills + ["retrieval: memory"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    # Those must be overriden
    dataset: str = ""
    dataset_description: str = ""
    number_of_other_states: int = 0

    def get_state_data(
        self,
        states_df: pd.DataFrame,
        state: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> tuple[list[pd.Series], list[str]]:
        """
        For the correct state, and number_of_other_states other ones, return their values.
        The states are shuffled before being returned.
        """
        other_states = list(states_df.name.unique())
        other_states.remove(state)
        selection = self.random.choice(
            other_states, size=self.number_of_other_states, replace=False
        ).tolist()
        selection = selection + [state]
        # shuffle is an in-place operation
        self.random.shuffle(selection)

        state_data = [
            states_df[
                (states_df.name == s)
                & (states_df.date >= window_start)
                & (states_df.date <= window_end)
            ].set_index("date")["value"]
            for s in selection
        ]
        return state_data, selection

    def random_instance(self):
        counties_df, states_df = download_fred_dataset(self.dataset)

        # Hardcoded 6 months of each, to only have 2024 data in the forecasting window
        history_start = pd.Timestamp("2023-08-01")
        history_end = pd.Timestamp("2024-01-01")
        forecast_start = pd.Timestamp("2024-02-01")
        forecast_end = pd.Timestamp("2024-07-01")

        # Many counties share names, we get the list of all (county name, state), and count how often
        # each county name show up. We then only pick from those who are unique.
        # This avoid ambiguities where the model could be confused about which county the data is about.
        names_freq = (
            counties_df[["name", "state"]].drop_duplicates()["name"].value_counts()
        )
        county = self.random.choice(names_freq[names_freq == 1].index)

        county_df = counties_df[
            (counties_df.name == county)
            & (counties_df.date >= history_start)
            & (counties_df.date <= forecast_end)
        ]
        state = county_df["state"].iloc[0]
        county_df = county_df.set_index("date")[["value"]]
        state_data, state_names = self.get_state_data(
            states_df, state, window_start=history_start, window_end=forecast_end
        )

        # Instantiate the class variables
        self.past_time = county_df[
            (county_df.index >= history_start) & (county_df.index <= history_end)
        ]
        self.future_time = county_df[
            (county_df.index >= forecast_start) & (county_df.index <= forecast_end)
        ]
        self.constraints = None
        self.background = self.get_background(county, state)
        self.scenario = self.get_scenario(state_data, state_names)

    def get_background(self, county: str, state: str) -> str:
        return f"This is the {self.dataset_description} for {county}, in the USA."

    def get_scenario(self, state_data: list[pd.Series], state_names: list[str]) -> str:
        if self.number_of_other_states == 0:
            intro = f"For reference, here is the {self.dataset_description} for an American state during the same period:"
        else:
            intro = f"For reference, here is the {self.dataset_description} for a few American states during the same period:"

        entries = []
        for data, name in zip(state_data, state_names):
            entry = name + "\n"
            entry += "-" * 20 + "\n"
            entry += write_state_series(data)
            entries.append(entry)
        return intro + "\n" + "\n\n".join(entries)

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return -1


class UnemploymentCountyUsingSingleStateData(BaseFREDCountyUsingStateData):
    dataset: str = "unemployment"
    dataset_description: str = "Unemployment Rate"
    number_of_other_states: int = 0
    __version__ = "0.0.1"  # Modification will trigger re-caching


class UnemploymentCountyUsingMultipleStateData(BaseFREDCountyUsingStateData):
    # It tests whether the model has memorized in which state is the county in
    _skills = BaseFREDCountyUsingStateData._skills + [
        "retrieval: context",
        "reasoning: analogy",
    ]
    dataset: str = "unemployment"
    dataset_description: str = "Unemployment Rate"
    number_of_other_states: int = 2
    __version__ = "0.0.1"  # Modification will trigger re-caching


class UnemploymentCountyUsingExplicitMultipleStateData(BaseFREDCountyUsingStateData):
    _skills = BaseFREDCountyUsingStateData._skills + [
        "retrieval: context",
        "reasoning: analogy",
    ]
    dataset: str = "unemployment"
    dataset_description: str = "Unemployment Rate"
    number_of_other_states: int = 2
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background(self, county: str, state: str) -> str:
        return f"This is the {self.dataset_description} for {county}, in {state}."


__TASKS__ = [
    UnemploymentCountyUsingSingleStateData,
    UnemploymentCountyUsingMultipleStateData,
    UnemploymentCountyUsingExplicitMultipleStateData,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            UnemploymentCountyUsingSingleStateData,
            UnemploymentCountyUsingMultipleStateData,
            UnemploymentCountyUsingExplicitMultipleStateData,
        ],
    ),
]
