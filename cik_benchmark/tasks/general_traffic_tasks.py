"""
Tasks based on the Monash `traffic` dataset
"""

import datetime
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pathlib import Path

from typing import Optional, Union, List
from abc import abstractmethod
from functools import partial
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository import get_dataset

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from . import WeightCluster
from .task_utils import convert_to_arrow

# TODO: rename to traffic_holiday_tasks.py
from ..data.pems import load_traffic_series

# from .ts_reasoner import TSReasoner

get_dataset = partial(get_dataset, path=DATA_STORAGE_PATH)


class TrafficTask(UnivariateCRPSTask):
    """
    Forecasting task based on the Monash traffic dataset or PEMS data after 01/01/2024 (if fresh_data==True)
    with frequency hourly, where the history is 7 days (168 timesteps) and prediction is 3 days (72 timesteps),
    and windows are chosen such that there is a holiday at the beginning of the prediction window.
    """
    __version__ = "0.0.1"  # Modification will trigger re-caching
    __name__ = "traffic"

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None, fresh_data=True):
        self.fresh_data = fresh_data
        self.history_length = 24 * 7  # 7 days
        self.future_length = 24 * 3   # 3 days
        self.holidays = self._get_holidays()
        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_holidays(self):
        """Return the list of holidays based on the data freshness."""
        if self.fresh_data:
            return [
                (datetime.date(2024, 1, 1), "New Year's Day"),
                (datetime.date(2024, 1, 15), "Martin Luther King Jr. Day"),
                (datetime.date(2024, 2, 19), "Washington's Birthday"),
                (datetime.date(2024, 5, 27), "Memorial Day"),
                (datetime.date(2024, 6, 19), "Juneteenth National Independence Day"),
                (datetime.date(2024, 7, 4), "Independence Day"),
                (datetime.date(2024, 9, 2), "Labor Day"),
                (datetime.date(2024, 10, 14), "Columbus Day"),
                (datetime.date(2024, 11, 11), "Veterans Day"),
                (datetime.date(2024, 11, 28), "Thanksgiving Day"),
                (datetime.date(2024, 12, 25), "Christmas Day"),
            ]
        else:
            return [
                (datetime.date(2015, 5, 25), "Memorial Day"),
                (datetime.date(2015, 7, 4), "Independence Day"),
                (datetime.date(2015, 9, 7), "Labor Day"),
                (datetime.date(2015, 11, 11), "Veterans Day"),
                (datetime.date(2015, 11, 26), "Thanksgiving"),
                (datetime.date(2015, 12, 25), "Christmas Day"),
                (datetime.date(2016, 5, 30), "Memorial Day"),
                (datetime.date(2016, 7, 4), "Independence Day"),
                (datetime.date(2016, 9, 5), "Labor Day"),
                (datetime.date(2016, 11, 11), "Veterans Day"),
                (datetime.date(2016, 11, 24), "Thanksgiving"),
                (datetime.date(2016, 12, 25), "Christmas Day"),
            ]

    def save_dataset_to_arrow(self, path: Union[str, Path], compression: str = "lz4"):
        """
        Load dataset and save it in Apache Arrow format using convert_to_arrow.

        :param path: File path where the Arrow dataset will be stored.
        :param compression: Compression format for Arrow file (default is "lz4").
        """
        dataset = self._get_dataset_if_needed()
        if dataset is None:
            raise ValueError("Dataset could not be retrieved. Ensure fresh_data is set to False.")

        # Extract time series data from the dataset
        time_series = [to_pandas(series).values for series in dataset.train]

        # Use the existing convert_to_arrow function
        convert_to_arrow(path, time_series, compression=compression)

    def get_series(self, dataset_name: str = "traffic", target=None):
        if dataset_name == "traffic":
            target = target or "Occupancy (%)"
            return load_traffic_series(target=target, random=self.random)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    def _prepare_full_series(self, full_series):
        """Clean the time series by ensuring a proper datetime index, dropping incomplete days,
        and aligning the start to a Monday."""
        if isinstance(full_series.index, pd.PeriodIndex):
            full_series.index = full_series.index.to_timestamp()
        full_series.index = pd.to_datetime(full_series.index)
        first_ts = full_series.index[0]
        if first_ts.time() != pd.Timestamp("00:00:00").time():
            next_midnight = first_ts.normalize() + pd.Timedelta(days=1)
            full_series = full_series.loc[next_midnight:]
        # Align start to Monday (Monday=0)
        first_day = full_series.index[0].day_of_week
        if first_day != 0:
            full_series = full_series.iloc[(7 - first_day) * 24:]
        return full_series

    def _get_dataset_if_needed(self):
        """Retrieve dataset when fresh_data is False."""
        if not self.fresh_data:
            dataset = get_dataset("traffic", regenerate=False)
            assert len(dataset.train) == len(dataset.test), "Train and test sets must contain the same number of time series"
            return dataset
        return None

    def sample_holiday_instance(self):
        """
        Select a series where the holiday window shows a significant drop in traffic.
        Returns history, future series and the holiday details.
        """
        dataset = self._get_dataset_if_needed()
        window_is_interesting = False
        while not window_is_interesting:
            if self.fresh_data:
                full_series = self.get_series("traffic")
            else:
                ts_index = self.random.choice(len(dataset.test))
                full_series = to_pandas(list(dataset.test)[ts_index])
            full_series = self._prepare_full_series(full_series)

            # Pick a random holiday
            holiday_date, holiday_name = self.holidays[self.random.choice(len(self.holidays))]
            holiday_dt = pd.to_datetime(holiday_date)
            try:
                holiday_index = full_series.index.get_loc(holiday_dt)
            except KeyError:
                continue  # holiday not in series; try again

            # Define history and future windows
            history_series = full_series.iloc[holiday_index - (24 * 7):holiday_index]
            future_series = full_series.iloc[holiday_index:holiday_index + (24 * 3)]
            holiday_series = full_series.iloc[holiday_index:holiday_index + 24]

            # Check for significant drop in holiday traffic
            if holiday_series.mean() <= 0.7 * history_series.mean():
                window_is_interesting = True

        return history_series, future_series, holiday_date, holiday_name

    def sample_random_instance(self):
        """
        Randomly sample a valid time series instance.
        Returns history and future series.
        """
        dataset = self._get_dataset_if_needed()
        if self.fresh_data:
            full_series = self.get_series("traffic")
        else:
            ts_index = self.random.choice(len(dataset.test))
            full_series = to_pandas(list(dataset.test)[ts_index])
        full_series = self._prepare_full_series(full_series)

        if len(full_series) < self.history_length + self.future_length:
            raise ValueError("full_series is too short for the required sampling.")

        max_start_index = len(full_series) - (self.history_length + self.future_length)
        start_index = self.random.randint(0, max_start_index)
        history_series = full_series.iloc[start_index : start_index + self.history_length]
        future_series = full_series.iloc[start_index + self.history_length : start_index + self.history_length + self.future_length]
        return history_series, future_series

    def _initialize_instance(self, history_series, future_series, background, roi=slice(0, 24)):
        """Common initialization of instance variables."""
        self.background = background
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.scenario = None
        self.region_of_interest = roi

    @abstractmethod
    def get_background(self, future_series=None, holiday_date=None, holiday_name=None):
        """Generate a textual hint for the model regarding the forecast context."""
        pass


class TrafficTask_Random(TrafficTask):
    """
    Randomly sample a traffic instance for forecasting.
    """
    __version__ = "0.0.1"

    def random_instance(self):
        history_series, future_series = self.sample_random_instance()
        background = self.get_background(future_series)
        self._initialize_instance(history_series, future_series, background)

    def get_background(self, future_series=None, holiday_date=None, holiday_name=None):
        return ("This time series records hourly freeway traffic occupancy rates (in %). It exhibits strong daily seasonality (24-hour cycles) and weekly seasonality (168-hour cycles), with notably lower occupancy during weekends and holidays.")


class TrafficTask_Holiday(TrafficTask):
    """
    Sample a traffic instance starting at a holiday, highlighting the impact of the holiday.
    """
    __version__ = "0.0.1"

    def random_instance(self):
        history_series, future_series, holiday_date, holiday_name = self.sample_holiday_instance()
        background = self.get_background(future_series, holiday_date, holiday_name)
        self._initialize_instance(history_series, future_series, background)

    def get_background(self, future_series=None, holiday_date=None, holiday_name=None):
        background = ("This series contains the road occupancy rates on a freeway in the San Francisco Bay area. "
                      f"Note that {holiday_date} is a holiday due to {holiday_name}. "
                      "Traffic on this freeway typically reduces on weekends and holidays.")
        return background


__TASKS__ = [
    TrafficTask_Random,
    # TrafficTask_Holiday
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            TrafficTask_Random,
            TrafficTask_Holiday
        ],
    ),
]
