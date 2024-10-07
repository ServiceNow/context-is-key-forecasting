import numpy as np

from functools import partial
from gluonts.dataset.util import to_pandas
from tactis.gluon.dataset import get_dataset

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str


get_dataset = partial(get_dataset, path=DATA_STORAGE_PATH)


class PredictableSpikesInPredTask(UnivariateCRPSTask):
    """
    Adds spikes to an arbitrary series.
    The presence of the spike is included in the context.
    Time series: agnostic
    Context: synthetic
    Parameters:
    ----------
    fixed_config: dict
        A dictionary with fixed parameters for the task
    seed: int
        Seed for the random number generator
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def random_instance(self):
        datasets = ["electricity_hourly"]

        # Select a random dataset
        dataset_name = self.random.choice(datasets)
        dataset = get_dataset(dataset_name, regenerate=False)

        assert len(dataset.train) == len(
            dataset.test
        ), "Train and test sets must contain the same number of time series"

        # Get the dataset metadata
        metadata = dataset.metadata

        # Select a random time series
        ts_index = self.random.choice(len(dataset.train))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=self.random.randint(2, 5),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        # Arbitrary way to select a spike date: sort the values of future_series (excluding the last point), pick it from the largest 5 values
        spike_idx = self.random.choice(np.argsort(future_series.values[:-1])[-5:][::-1])
        spike_datetime = future_series.index[spike_idx]

        history_series.index = history_series.index.to_timestamp()
        future_series.index = future_series.index.to_timestamp()
        ground_truth = future_series.copy()

        relative_impact = self.random.randint(1, 500)
        is_negative = self.random.choice([True, False])
        if is_negative:
            relative_impact = -relative_impact

        future_series.iloc[spike_idx] *= np.float32(1 + relative_impact / 100)

        scenario = f"A fluctuation of {relative_impact}% is expected to affect the usual value of the series at exactly {datetime_to_str(spike_datetime)}, after which the series will return to normal."

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.ground_truth = ground_truth
        self.constraints = None
        self.background = None
        self.scenario = scenario

        # ROI metric parameters
        self.region_of_interest = int(spike_idx)


__TASKS__ = [PredictableSpikesInPredTask]
