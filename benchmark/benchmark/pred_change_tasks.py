import numpy as np

from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from .base import UnivariateCRPSTask
from .utils import get_random_window_univar


class DecreaseInTrafficInPrediction(UnivariateCRPSTask):
    """
    A task where the traffic was lower than usual in prediction part,
    due to an accident.
    This should be deducted from the context and reflected in the forecast.
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = ["traffic"]

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
            history_factor=self.random.randint(3, 7),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        # Decide the drop duration/start date/magnitude and introduce the drop
        future_series.index = future_series.index.to_timestamp()
        drop_duration = self.random.choice(
            [1, 2, 3, 4, 5, 6, 7]
        )  # Arbitrarily picked from 1-7 hours
        drop_start_date = self.random.choice(
            future_series.index[
                :-8
            ]  # Starting point is anywhere from start of series to max(drop_duration) + 1 points before the series. +1 is arbitrary, to have the drop not completely at the end of the pred.
        )  # Arbitrary start point for now
        drop_start_point = future_series.index.get_loc(drop_start_date)
        drop_magnitude = self.random.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5]
        )  # Arbitrarily set to 0.1 to 0.5 times the usual value in the time series
        # Add drop to the data
        future_series.iloc[drop_start_point : drop_start_point + drop_duration] = (
            drop_magnitude * future_series[drop_start_point]
        )

        # Convert future index to timestamp for consistency
        history_series.index = history_series.index.to_timestamp()

        background = f"This is hourly traffic data."
        scenario = f"Consider that there was an accident on the road and there was {drop_magnitude*100}% of the usual traffic from {drop_start_date} for {drop_duration} hours."  # TODO: May also specify drop end date instead of the drop duration.

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = scenario


__TASKS__ = [DecreaseInTrafficInPrediction]