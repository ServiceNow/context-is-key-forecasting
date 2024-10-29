import numpy as np

from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str
from . import WeightCluster

from benchmark.data.pems import (
    load_traffic_series,
    get_traffic_prediction_length,
    get_traffic_history_factor,
)


class DecreaseInTrafficInPredictionTask(UnivariateCRPSTask):
    """
    A task where the traffic was lower than usual in prediction part,
    due to an accident.
    This should be deducted from the context and reflected in the forecast.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def random_instance(self):
        datasets = ["traffic_fresh"]
        dataset_name = self.random.choice(datasets)

        if dataset_name == "traffic":
            # Select a random dataset

            dataset = get_dataset(
                dataset_name, regenerate=False, path=DATA_STORAGE_PATH
            )

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

            # Convert future index to timestamp for consistency
            history_series.index = history_series.index.to_timestamp()
            future_series.index = future_series.index.to_timestamp()

        elif dataset_name == "traffic_fresh":

            prediction_length = get_traffic_prediction_length()

            max_attempts = 100
            for _ in range(max_attempts):
                target = "Occupancy (%)"
                full_series = load_traffic_series(target=target, random=self.random)

                try:
                    # Select a random window
                    window = get_random_window_univar(
                        full_series,
                        prediction_length=prediction_length,
                        history_factor=get_traffic_history_factor(),
                        random=self.random,
                        max_attempts=1,  # Handle the attempts in this method instead
                    )
                    break
                except ValueError:
                    # This exception is thrown if get_random_window_univar did not select a valid window
                    pass
            else:
                raise ValueError(
                    f"Could not find a valid window after {max_attempts} attempts"
                )

            # Extract the history and future series
            history_series = window.iloc[:-prediction_length]
            future_series = window.iloc[-prediction_length:]

        # Decide the drop duration/start date/magnitude and introduce the drop
        drop_duration = self.random.choice(
            [1, 2, 3, 4, 5, 6, 7]
        )  # Arbitrarily picked from 1-7 hours
        # Arbitrary way to select a start date: sort the values of future_series (excluding the last drop_duration+1 points), pick it from the largest 5 values
        drop_start_point = self.random.choice(
            np.argsort(future_series.values[: -(drop_duration + 1)])[-5:][::-1]
        )
        drop_start_date = future_series.index[drop_start_point]
        drop_magnitude = self.random.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5]
        )  # Arbitrarily set to 0.1 to 0.5 times the usual value in the time series
        # Add drop to the data
        future_series.iloc[drop_start_point : drop_start_point + drop_duration] = (
            drop_magnitude * future_series[drop_start_point]
        )

        background = f"This is hourly traffic data."
        scenario = f"Suppose that there is an accident on the road and there is {drop_magnitude*100}% of the usual traffic from {datetime_to_str(drop_start_date)} for {drop_duration} {'hours' if drop_duration > 1 else 'hour'}."  # TODO: May also specify drop end date instead of the drop duration.

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = scenario

        # ROI metric parameters
        self.region_of_interest = slice(
            drop_start_point, drop_start_point + drop_duration
        )


__TASKS__ = [DecreaseInTrafficInPredictionTask]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            DecreaseInTrafficInPredictionTask,
        ],
    ),
]
