from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas
import numpy as np
import pandas as pd

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str
from ..memorization_mitigation import add_realistic_noise
from . import WeightCluster

from typing import Optional
from abc import abstractmethod


class ElectricityTask(UnivariateCRPSTask):
    """
    A task where the consumption of electricity spikes in prediction part,
    due to a heat wave and people using a lot of air conditioning.
    The spikes should be deducted from the context and reflected in the forecast.
    TODO: A multivariate extension of this task, where weather is another time series

    """
    __version__ = "0.0.1"  # Modification triggers re-caching
    __name__ = "electricity"

    def __init__(
        self,
        seed: int = None,
        fixed_config: Optional[dict] = None,
    ):
        self._dataset_cache = None
        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_dataset_if_needed(self):
        """
        Load or retrieve the dataset(s). 
        You can handle 'fresh_data' logic here if needed.
        """
        if self._dataset_cache is None:
            # If your code only ever uses "electricity_hourly", 
            # just load that directly
            dataset = get_dataset(
                "electricity_hourly", regenerate=False, path=DATA_STORAGE_PATH
            )
            self._dataset_cache = {"electricity_hourly": dataset}
        return self._dataset_cache

    def sample_random_instance(self):
        """
        Encapsulates the logic previously in `random_instance`. 
        - Picks dataset "electricity_hourly" (or random if you have multiple).
        - Selects a random time series, a random window, and inserts a spike in consumption.
        - Adds noise via `make_transform()`.
        Returns (history_series, future_series, background, scenario, roi).
        """
        # 1) Get or cache the dataset(s)
        datasets = self._get_dataset_if_needed()
        dataset_name = self.random.choice(list(datasets.keys()))  # e.g. "electricity_hourly"
        dataset = datasets[dataset_name]

        assert len(dataset.train) == len(dataset.test), (
            "Train and test sets must contain the same number of time series"
        )

        metadata = dataset.metadata

        # 2) Select a random time series
        ts_index = self.random.choice(len(dataset.train))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # 3) Select a random window (history + future) from the time series
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=self.random.randint(3, 7),
            random=self.random,
        )
        # Extract the history and future
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        # 4) Insert a spike in the future portion 
        #    (heat wave scenario leading to increased consumption).
        #    For example: pick a random start among the top values, 
        #    choose a spike duration and magnitude, then multiply.
        future_series.index = future_series.index.to_timestamp()  # ensure Timestamp index

        # Convert history index to timestamp for consistency
        history_series.index = history_series.index.to_timestamp()

        # 5) Transform (add noise, constraints, etc.)
        history_series, future_series = self.make_transform(
            history_series, future_series
        )

        # 6) Build a 'background' and/or scenario
        #    In your original code, background is a short text, scenario is more specific.

        return history_series, future_series

    def make_transform(self, history_series, future_series):
        """
        Apply a 'realistic noise' transform or any other transformations 
        you want to apply to both history and future.
        """
        history_series = add_realistic_noise(history_series, self.random)
        future_series = add_realistic_noise(future_series, self.random)
        return history_series, future_series

    @abstractmethod
    def get_background(self, future_series=None, holiday_date=None, holiday_name=None):
        """Generate a textual hint for the model regarding the forecast context."""
        pass

    def _initialize_instance(
        self,
        history_series: pd.Series,
        future_series: pd.Series,
        background: str,
        roi: slice = None,
    ):
        """
        Common initializer to set instance variables, 
        similar to TrafficTask.
        """
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.background = background
        self.scenario = None
        self.constraints = None
        self.region_of_interest = roi

    @property
    def seasonal_period(self) -> int:
        """
        If the entire daily period isn't reliably covered, 
        we might set -1 to indicate 'no period' for classical models.
        """
        return -1

    @property
    def max_directprompt_batch_size(self) -> int:
        """
        If using direct prompting, limit batch size to avoid OOM issues 
        on large LLM servers.
        """
        return 5


class ElectricityTask_Random(ElectricityTask):
    """
    Subclass that follows the 'Random' pattern: it just calls 
    the base class's sample method and initializes the instance.
    """

    __version__ = "0.0.1"

    def random_instance(self):
        """
        Similar to TrafficTask_Random, we:
          1) sample a random instance
          2) call _initialize_instance with the results
        """
        history_series, future_series = self.sample_random_instance()

        background = self.get_background(future_series)

        self._initialize_instance(
            history_series=history_series,
            future_series=future_series,
            background=background,
        )

    def get_background(self, future_series=None):
        return ("This is the electricity consumption recorded in Kilowatt (kW) in city A.")


__TASKS__ = [
    ElectricityTask_Random,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            ElectricityTask_Random,
        ],
    ),
]
