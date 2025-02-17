"""
Tasks using series on FRED which are available at both county and state levels.

https://fred.stlouisfed.org
"""

import pandas as pd

from ..base import UnivariateCRPSTask
from . import WeightCluster

from .fred_county_tasks import download_fred_dataset, write_state_series

from abc import abstractmethod
import pandas as pd
from typing import Optional, List, Tuple

# Constants for date slicing
HISTORY_START = pd.Timestamp("2023-08-01")
HISTORY_END   = pd.Timestamp("2024-01-01")
FORECAST_START= pd.Timestamp("2024-02-01")
FORECAST_END  = pd.Timestamp("2024-07-01")


class FredTask(UnivariateCRPSTask):
    """
    Base class for a FRED-based monthly-forecasting task,
    where the model sees limited county-level history and
    also sees a reference state-level (or multiple states) series
    for the same period.
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching
    __name__ = "fred"

    # Override these at runtime or in subclasses
    dataset: str = ""
    dataset_description: str = ""    # e.g. "unemployment rate" or "housing starts"
    number_of_other_states: int = 0  # how many extra states to provide

    def __init__(
        self,
        seed: int = None,
        fixed_config: Optional[dict] = None,
    ):
        self._fred_data = None  # Will cache (counties_df, states_df) after download
        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_dataset_if_needed(self):
        """
        Download or retrieve the FRED dataset only once.
        If 'fresh_data' is True, you could add logic to force re-download.
        """
        if self._fred_data is None:
            self._fred_data = download_fred_dataset(self.dataset)
        return self._fred_data

    def get_state_data(
        self,
        states_df: pd.DataFrame,
        state: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> Tuple[List[pd.Series], List[str]]:
        """
        Returns the main state's series plus 'number_of_other_states' more states,
        all sliced for [window_start..window_end].
        Shuffles the final list so the correct state is not obviously last.
        """
        unique_states = list(states_df.name.unique())
        unique_states.remove(state)

        # Pick random states, plus the main state
        selection = self.random.choice(
            unique_states,
            size=self.number_of_other_states,
            replace=False
        ).tolist() + [state]

        self.random.shuffle(selection)

        state_data = [
            (
                states_df.loc[
                    (states_df.name == s) & (states_df.date.between(window_start, window_end)),
                    ["date", "value"]
                ]
                .set_index("date")["value"]  # Now "date" is the index, we extract the "value" column as a Series
            )
            for s in selection
        ]

        return state_data, selection

    def sample_random_instance(self):
        """
        Core logic for selecting a unique county, slicing its monthly data
        into history vs. future, and retrieving reference data for the state(s).
        """
        # 1) Load data
        counties_df, states_df = self._get_dataset_if_needed()

        # 2) Identify counties with unique names (avoids confusion in multi-state duplicates)
        names_freq = counties_df[["name", "state"]].drop_duplicates()["name"].value_counts()
        unique_counties = names_freq[names_freq == 1].index
        if not len(unique_counties):
            raise ValueError("No counties have uniquely identifying names in this dataset.")

        # 3) Randomly pick a unique county
        county = self.random.choice(unique_counties)

        # 4) Slice data for [HISTORY_START..FORECAST_END]
        county_df = counties_df.loc[
            (counties_df.name == county)
            & (counties_df.date.between(HISTORY_START, FORECAST_END))
        ]
        if county_df.empty:
            raise ValueError(f"No data found for {county} in the specified window.")

        # 5) Retrieve the state from the chosen county row
        state = county_df["state"].iloc[0]

        # 6) Convert to monthly time-indexed DataFrame
        county_df = county_df.set_index("date")[["value"]]

        # 7) Gather state data for [HISTORY_START..FORECAST_END]
        state_data, state_names = self.get_state_data(
            states_df=states_df,
            state=state,
            window_start=HISTORY_START,
            window_end=FORECAST_END
        )

        # 8) Slice past/future
        past_time = county_df.loc[HISTORY_START: HISTORY_END]
        future_time = county_df.loc[FORECAST_START: FORECAST_END]

        # Return all components needed for final initialization
        return past_time, future_time, county, state, state_data, state_names

    @abstractmethod
    def get_background(self, county: str, state: str) -> str:
        """Generate a textual hint for the model regarding the forecast context."""
        pass

    def _initialize_instance(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        background: str,
        roi = None
    ):
        """
        Configure instance variables for the task.
        """
        self.past_time = past_time
        self.future_time = future_time
        self.background = background
        self.scenario = None
        self.constraints = None
        self.region_of_interest = roi

    @property
    def seasonal_period(self) -> int:
        """
        Return -1 if we don't rely on a known period, or set an integer for monthly seasonality.
        """
        return -1


class FredTask_Random(FredTask):
    """
    Subclass for random sampling of the FRED data, akin to TrafficTask_Random.
    """

    __version__ = "0.0.1"

    dataset: str = "unemployment"
    dataset_description: str = "Unemployment Rate"
    number_of_other_states: int = 2

    def random_instance(self):
        # 1) Generate slices & metadata from the base class
        (
            past_time,
            future_time,
            county,
            state,
            state_data,
            state_names,
        ) = self.sample_random_instance()

        # 2) Build textual background and scenario
        background = self.get_background(county, state)

        # 3) Initialize the instance
        self._initialize_instance(
            past_time=past_time,
            future_time=future_time,
            background=background,
        )

    def get_background(self, county: str, state: str) -> str:
        """
        A short textual description about the county & dataset.
        """
        return (
            f"This is the {self.dataset_description} for {county}, "
            f"in {state} (USA). Data is monthly, with limited history."
        )


__TASKS__ = [
    FredTask_Random,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            FredTask_Random,
        ],
    ),
]
