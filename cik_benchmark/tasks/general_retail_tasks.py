"""
Tasks based on the NN5 dataset (nn5_daily_without_missing), which is a dataset (with 111 series) of total number of cash withdrawals from 111 different Automated Teller
Machines (ATM) in the UK.
"""
from typing import Optional

from functools import partial
from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas
import numpy as np
import pandas as pd

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str
from ..memorization_mitigation import add_realistic_noise
from . import WeightCluster

from abc import abstractmethod

get_dataset = partial(get_dataset, path=DATA_STORAGE_PATH)


class RetailTask(UnivariateCRPSTask):
    """
    Base class for a retail (ATM withdrawals) forecasting task where 
    the ATM is depleted of cash for a duration in the prediction horizon, 
    causing zero withdrawals in that period.
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self, 
        seed: int = None, 
        fixed_config: Optional[dict] = None,
    ):
        self._dataset_cache = None  # Will hold loaded datasets
        # You can store references to possible dataset names here if needed:
        self._possible_datasets = ["nn5_daily_without_missing"]
        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_dataset_if_needed(self, dataset_name: str):
        """
        Load or retrieve the dataset by name, caching it 
        so we don't reload multiple times.
        """
        if self._dataset_cache is None:
            self._dataset_cache = {}
        if dataset_name not in self._dataset_cache:
            # Example: get_dataset(...)
            ds = get_dataset(dataset_name, regenerate=False)
            self._dataset_cache[dataset_name] = ds
        return self._dataset_cache[dataset_name]

    def sample_random_instance(self):
        """
        1) Pick a dataset at random from self._possible_datasets.
        2) Select a random time series and random window from the dataset.
        3) Introduce a 'drop' period in the future portion where withdrawals = 0.
        4) Add realistic noise to both history & future.
        5) Build a background note & scenario text.
        6) Return everything needed to initialize the task instance.
        """
        # 1) Choose the dataset
        dataset_name = self.random.choice(self._possible_datasets)
        dataset = self._get_dataset_if_needed(dataset_name)

        assert len(dataset.train) == len(dataset.test), (
            "Train and test sets must contain the same number of time series."
        )
        metadata = dataset.metadata

        # 2) Select a random time series & slice a random window
        ts_index = self.random.choice(len(dataset.train))
        full_series = to_pandas(list(dataset.test)[ts_index])

        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=self.random.randint(2, 4),
            random=self.random,
        )
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        return history_series, future_series

    def _initialize_instance(
        self,
        history_series: pd.Series,
        future_series: pd.Series,
        background: str = None,
        scenario: str = None,
        roi: Optional[slice] = None
    ):
        """
        Set up the base instance variables and constraints, similar to TrafficTask.
        """
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = background
        self.scenario = None
        self.constraints = None
        self.region_of_interest = roi

    @abstractmethod
    def get_background(self, drop_duration, drop_spacing, drop_start_date):
        """Generate a textual hint for the model regarding the forecast context."""
        pass

    @property
    def seasonal_period(self) -> int:
        """
        Return -1 if the horizon doesn't match a known seasonal period.
        """
        return -1


class RetailTask_Random(RetailTask):
    """
    Subclass that calls sample_random_instance() to get a random ATM-withdrawals
    scenario, then initializes the instance. Similar to TrafficTask_Random.
    """

    __version__ = "0.0.1"

    def random_instance(self):
        """
        1) Sample data from the base class
        2) Initialize the instance with that data
        """
        history_series, future_series = self.sample_random_instance()

        background = self.get_background()

        self._initialize_instance(
            history_series=history_series,
            future_series=future_series,
            background=background,
        )

    def get_background(self, drop_duration=None, drop_spacing=None, drop_start_date=None):
        return (f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England.")


__TASKS__ = [
    RetailTask_Random,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            RetailTask_Random,
        ],
    ),
]