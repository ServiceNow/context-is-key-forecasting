"""
Tasks where the given history is too short to accurately do the forecasting,
but context is given to allow the model to fill in the missing information.
"""

import datetime
import pandas as pd
from abc import abstractmethod

from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository import get_dataset

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..metrics.constraints import MinConstraint
from .nsrdb_tasks import download_all_nsrdb_datasets
from ..memorization_mitigation import add_realistic_noise
from . import WeightCluster

from typing import Optional, Union


class ClimatologyIrradianceTask(UnivariateCRPSTask):
    """
    We use NSRDB data at 10-minute intervals for the year 2022, focusing on
    Direct Normal Irradiance (DNI), which tends to be more variable than GHI.
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching
    __name__ = "climatology_irradiance"

    def __init__(
        self,
        seed: int = None,
        fixed_config: Optional[dict] = None,
    ):
        """
        :param seed: Random seed for reproducibility.
        :param fixed_config: Optional config dictionary for the task.
        :param fresh_data: If True, you could implement logic to download
                           the newest dataset. For now, used as a toggle
                           or simply unused if always fetching the same data.
        """
        # You can define history/future lengths if you like; for demonstration:
        self.num_ints_perday = 24 * 6
        self.history_length = 12 * 6  # 24 hours * 6 intervals/hour = 144 intervals
        self.future_length = 12 * 6   # same for the next 24 hours
        self._all_data = None         # Will hold the cached dataset, if desired
        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_dataset_if_needed(self):
        """
        Fetches and optionally caches the NSRDB dataset.
        Mimics the approach in TrafficTask where data retrieval
        is done in _get_dataset_if_needed.
        """
        if self._all_data is None:
            # This call should return a list of time series + headers, e.g.:
            # all_data = [
            #   (header_dict, df_of_variables), ...
            # ]
            self._all_data = download_all_nsrdb_datasets(interval=10)
        return self._all_data

    def sample_random_instance(self):
        """
        Perform the random selection from NSRDB data

        Returns:
          history_series, future_series, full_history_series, state, country
        """

        # Get data (downloading only once if needed)
        all_data = self._get_dataset_if_needed()

        # Select a random time series
        ts_index = self.random.choice(len(all_data))
        header = all_data[ts_index][0]
        # Use DNI based on your docstring note:
        full_series = all_data[ts_index][1]["DNI"]

        N = len(full_series)

        # Ensure you can fit both history_length and future_length
        max_start = N - (self.history_length + self.future_length)
        if max_start <= 0:
            raise ValueError("Not enough data to sample both history and future.")

        # Pick a random start index
        start_idx = self.random.randint(0, max_start + 1)

        # Slice the time series into history and future
        history_series = full_series.iloc[start_idx : start_idx + self.history_length]
        future_series = full_series.iloc[
            start_idx + self.history_length : start_idx + self.history_length + self.future_length
        ]
        full_history_series = full_series.iloc[:start_idx]

        # Example metadata from the header
        state = header["State"][2:-1]    # e.g. " TX "
        country = header["Country"][2:-1]

        return history_series, future_series, full_history_series, state, country

    def sample_halfday_instance(self):
        """
        Perform the random selection logic from NSRDB data, picking:
          - a random time series
          - a random day in the latter half of 2022
          - a random mid-day start for the forecast
        Returns:
          history_series, future_series, full_history_series, state, country
        """

        # Get data (downloading only once if needed)
        all_data = self._get_dataset_if_needed()

        # Select a random time series
        ts_index = self.random.choice(len(all_data))
        header = all_data[ts_index][0]
        # Use DNI based on your docstring note:
        full_series = all_data[ts_index][1]["DNI"]

        # The NSRDB dataset is for the whole year of 2022.
        # Select a day for the forecast between 2022-07-01 (day=181) and 2022-12-31 (day=365).
        day = self.random.randint(low=181, high=365)

        # Select the start of the forecast period, e.g. from 10:30 to 13:30 
        # (assuming 10-minute intervals => 6 intervals/hour, plus offset for minutes).
        forecast_time = self.random.randint(low=(10 * 6 + 3), high=(13 * 6 + 3) + 1)

        # Full history is everything before that day/time, so you can see the "year so far."
        full_history_series = full_series.iloc[: (day * self.num_ints_perday)]

        # History: from midnight of the chosen day up to forecast_time
        history_series = full_series.iloc[(day * self.num_ints_perday) : (day * self.num_ints_perday) + forecast_time]

        # Future: from forecast_time onward, up to the end of that same day
        future_series = full_series.iloc[
            (day * self.num_ints_perday) + forecast_time : (day * self.num_ints_perday) + self.num_ints_perday
        ]

        # Example metadata from the header
        state = header["State"][2:-1]    # e.g. " TX "
        country = header["Country"][2:-1]

        # Shift the day’s timestamps by 1 day forward, if desired:
        history_series.index = history_series.index + pd.Timedelta(days=1)
        future_series.index = future_series.index + pd.Timedelta(days=1)

        return history_series, future_series, full_history_series, state, country
    
    def _initialize_instance(
        self,
        history_series: pd.Series,
        future_series: pd.Series,
        background: str
    ):
        """
        Common initialization of instance variables, mimicking TrafficTask.
        """
        self.background = background
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.scenario = None
        # region_of_interest is not strictly needed here, but you could add one if desired
        self.region_of_interest = None

    @abstractmethod
    def get_background(
        self,
        full_history_series: pd.Series = None,
        state: str = None,
        country: str = None,
    ) -> str:
        """
        Generate a textual hint for the model about the daily solar intensity shape,
        location, or other relevant context.
        """
        pass

    @property
    def seasonal_period(self) -> int:
        """
        Returns the period which should be used by statistical models for this task.
        In this example, we simply set -1 to indicate the data is not providing a 
        full daily cycle in history (or that we don't want to rely on it).
        """
        return -1

    @property
    def max_directprompt_batch_size(self) -> int:
        """
        Only request up to this many samples at once when using direct prompting methods.
        """
        return 5


class ClimatologyIrradianceTask_Random(ClimatologyIrradianceTask):
    """
    Randomly sample a solar irradiance instance for forecasting,
    analogous to TrafficTask_Random.
    """

    __version__ = "0.0.1"

    def random_instance(self):
        """
        Samples a random instance (day/time) from the NSRDB data
        and initializes all relevant class variables.
        """
        (
            history_series,
            future_series,
            full_history_series,
            state,
            country,
        ) = self.sample_random_instance()

        background = self.get_background(
            full_history_series=full_history_series,
            state=state,
            country=country,
        )

        self._initialize_instance(history_series, future_series, background)

    def get_background(
        self,
        full_history_series: pd.Series = None,
        state: str = None,
        country: str = None,
    ) -> str:
        """
        Provide a textual note about the daily solar pattern, location, etc.
        This can help the model interpret the context (e.g., strong morning ramp-up,
        midday peak, evening drop, etc.).
        """
        return (
            f"This time series represents Direct Normal Irradiance (DNI) data for {state}, {country}. "
            "The signal follows a clear daily pattern, rising in the morning and dropping in the evening. "
            "Note that cloud cover and other local weather variations can influence the shape."
        )


class ClimatologyIrradianceTask_HalfDay(ClimatologyIrradianceTask):
    """
    Randomly sample a solar irradiance instance for forecasting,
    analogous to TrafficTask_Random.
    """

    __version__ = "0.0.1"

    def random_instance(self):
        """
        Samples a random instance (day/time) from the NSRDB data
        and initializes all relevant class variables.
        """
        (
            history_series,
            future_series,
            full_history_series,
            state,
            country,
        ) = self.sample_halfday_instance()

        background = self.get_background(
            full_history_series=full_history_series,
            state=state,
            country=country,
        )

        self._initialize_instance(history_series, future_series, background)

    def get_background(
        self,
        full_history_series: pd.Series = None,
        state: str  = None,
        country: str  = None,
    ) -> str:
        """
        Provide a textual note about the daily solar pattern, location, etc.
        This can help the model interpret the context (e.g., strong morning ramp-up,
        midday peak, evening drop, etc.).
        """
        return (
            f"This time series represents Direct Normal Irradiance (DNI) data for {state}, {country}. "
            "The signal follows a clear daily pattern, rising in the morning and dropping in the evening. "
            "Given only the first half of the day's irradiance, the model is asked to forecast the second half. "
            "Note that cloud cover and other local weather variations can influence the shape."
        )


class ClimatologyPowerProduction(UnivariateCRPSTask):
    """
    Base class for forecasting tasks involving solar power production data
    with a strong daily periodicity. The model is tasked to forecast the solar_10_minutes dataset (year 2006).
    """

    __version__ = "0.0.1"
    __name__ = "climatology_power_production"

    def __init__(
        self,
        seed: int = None,
        fixed_config: Optional[dict] = None,
    ):
        """
        :param seed: Random seed for reproducibility.
        :param fixed_config: Optional config dictionary for the task.
        """
        # These correspond to "the first 12 hours" and "the second 12 hours" 
        self.num_ints_per_day = 24 * 6
        self.history_length = 12 * 6  # 24 hours * 6 intervals/hour = 144 intervals
        self.future_length = 12 * 6   # same for the next 24 hours
        self._dataset = None
        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_dataset_if_needed(self):
        """
        Load or cache the solar_10_minutes dataset. 
        Equivalent to the pattern in TrafficTask.
        """
        if self._dataset is None:
            self._dataset = get_dataset(
                "solar_10_minutes", regenerate=False, path=DATA_STORAGE_PATH
            )
        return self._dataset

    def sample_random_instance(self):
        """
        Sample a random 24-hour window from the entire series, then choose a random
        split (forecast_time) inside that window. The part before forecast_time
        acts as history, and the part after as future.
        """

        # Load dataset if needed
        dataset = self._get_dataset_if_needed()
        ts_index = self.random.choice(len(dataset.test))
        full_series = to_pandas(list(dataset.test)[ts_index])

        N = len(full_series)

        # Ensure you can fit both history_length and future_length
        max_start = N - (self.history_length + self.future_length)
        if max_start <= 0:
            raise ValueError("Not enough data to sample both history and future.")

        # Pick a random start index
        start_idx = self.random.randint(0, max_start + 1)

        # Slice the time series into history and future
        history_series = full_series.iloc[start_idx : start_idx + self.history_length]
        future_series = full_series.iloc[
            start_idx + self.history_length : start_idx + self.history_length + self.future_length
        ]
        full_history_series = full_series.iloc[:start_idx]

        return history_series, future_series, full_history_series

    def sample_halfday_instance(self):
        """
        Internal method that encapsulates the existing random-instance logic.
        1) Retrieve the dataset
        2) Select a random time series
        3) Pick a day between 2006-07-01 (day=181) and 2006-12-31 (day=365)
        4) Pick a forecast time between 10:30 and 13:30
        5) Slice out history & future, add noise, and build background text
        6) Return (history_series, future_series, background)
        """

        dataset = self._get_dataset_if_needed()
        # Select a random time series from the test split
        ts_index = self.random.choice(len(dataset.test))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # Randomly pick a day and a forecast time window
        day = self.random.randint(low=181, high=365)  # 2006-07-01 to 2006-12-31
        forecast_time = self.random.randint(low=(10 * 6 + 3), high=(13 * 6 + 3) + 1)

        # For reference: day * 24 * 6 is the index offset for midnight of the chosen day
        start_of_day_idx = day * self.num_ints_per_day

        # full_history_series is everything before that day (not used in slicing below,
        # but you pass it to get_background)
        full_history_series = full_series.iloc[:start_of_day_idx]

        # history_series: from midnight of the chosen day up to 'forecast_time'
        history_series = full_series.iloc[start_of_day_idx : start_of_day_idx + forecast_time]

        # future_series: from 'forecast_time' to the end of that day (24h)
        future_series = full_series.iloc[
            start_of_day_idx + forecast_time : start_of_day_idx + self.num_ints_per_day
        ]

        history_series, future_series = self.make_transform(history_series, future_series)

        return history_series, future_series, full_history_series

    def _initialize_instance(
        self, 
        history_series: pd.Series, 
        future_series: pd.Series, 
        background: str
    ):
        """
        Common initialization method that sets the instance variables,
        analogous to TrafficTask.
        """
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = background
        self.constraints = None
        self.scenario = None
        # We can skip region_of_interest or set it to None
        # since you said the "full prediction window is important."
        self.region_of_interest = None

    @abstractmethod
    def get_background(
        self,
        full_history_series: pd.Series = None,
        future_start_time: Union[pd.Timestamp, None] = None,
    ) -> str:
        """
        Generate a textual hint to help the model interpret daily shape or context.
        e.g., "This is a solar power signal with typical dawn/dusk patterns."
        Must be implemented in subclasses.
        """
        pass

    def make_transform(self, history_series, future_series):
        """
        This separate transform method remains as in your code.
        If your pipeline expects it, you can keep it for manual transforms.
        """
        history_series = add_realistic_noise(
            history_series,
            self.random,
            noise_level=0.01,
            skip_zero_values=True,
            constraints=[MinConstraint(0)],
        )
        future_series = add_realistic_noise(
            future_series,
            self.random,
            noise_level=0.01,
            skip_zero_values=True,
            constraints=[MinConstraint(0)],
        )
        return history_series, future_series

    @property
    def seasonal_period(self) -> int:
        """
        Returns the period used by certain statistical models.
        In your original code, -1 indicates we are not applying a daily period.
        """
        return -1

    @property
    def max_directprompt_batch_size(self) -> int:
        """
        Restricts how many samples can be requested at once,
        if memory constraints are a concern.
        """
        return 5


class ClimatologyPowerProduction_Random(ClimatologyPowerProduction):
    """
    Subclass that follows the approach of TrafficTask_Random:
    it simply calls 'sample_random_instance()' and initializes 
    the instance variables.
    """

    __version__ = "0.0.1"

    def random_instance(self):
        """
        Public-facing method to retrieve a random instance,
        which calls into the base class logic.
        """
        history_series, future_series, full_history_series = self.sample_random_instance()

        background = self.get_background()

        self._initialize_instance(history_series, future_series, background)

    def get_background(
        self,
        full_history_series: pd.Series = None,
        future_start_time: Union[pd.Timestamp, None] = None,
    ) -> str:
        """
        Example background generator. You may copy in or adapt the logic from 
        your original 'get_background' method, or provide new text.
        """
        return (
            "This time series captures solar power production measured at 10-minute intervals during 2006. It exhibits a pronounced daily cycle, with low or near-zero output at night, a gradual ramp-up in the morning, and peak production around midday before declining toward evening. Variations in cloud cover, atmospheric conditions, and other local factors can introduce short-term fluctuations on top of this strong diurnal pattern."
        )
    

class ClimatologyPowerProduction_HalfDay(ClimatologyPowerProduction):
    """
    Subclass that follows the approach of TrafficTask_Random:
    it simply calls 'sample_random_instance()' and initializes 
    the instance variables.
    """

    __version__ = "0.0.1"

    def random_instance(self):
        """
        Public-facing method to retrieve a random instance,
        which calls into the base class logic.
        """
        history_series, future_series, full_history_series = self.sample_halfday_instance()

        background = self.get_background(
            full_history_series=full_history_series,
            future_start_time=future_series.index[0].start_time,
        )

        self._initialize_instance(history_series, future_series, background)

    def get_background(
        self,
        full_history_series: pd.Series,
        future_start_time: Union[pd.Timestamp, None],
    ) -> str:
        """
        Example background generator. You may copy in or adapt the logic from 
        your original 'get_background' method, or provide new text.
        """
        if future_start_time is not None:
            time_str = future_start_time.strftime("%H:%M")
        else:
            time_str = "N/A"

        return (
            "This is a solar power production time series (10-minute intervals) "
            "covering part of 2006. It exhibits strong daily seasonality, with "
            "low values overnight and higher output during daylight hours. "
            f"The forecast window starts around {time_str}, requiring an estimate "
            "of the power production for the remainder of the day."
        )
    

__TASKS__ = [
    ClimatologyIrradianceTask_Random,
    # ClimatologyIrradianceTask_HalfDay,
    ClimatologyPowerProduction_Random,
    # ClimatologyPowerProduction_HalfDay
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            ClimatologyIrradianceTask_Random,
            # ClimatologyIrradianceTask_HalfDay,
            ClimatologyPowerProduction_Random,
            # ClimatologyPowerProduction_HalfDay
        ],
    ),
]
