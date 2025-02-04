"""
Tasks based on the NN5 dataset (nn5_daily_without_missing), which is a dataset (with 111 series) of total number of cash withdrawals from 111 different Automated Teller
Machines (ATM) in the UK.
"""

from functools import partial
from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas
import numpy as np

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str
from ..memorization_mitigation import add_realistic_noise
from . import WeightCluster

get_dataset = partial(get_dataset, path=DATA_STORAGE_PATH)


class CashDepletedinATMScenarioTask(UnivariateCRPSTask):
    """
    This task considers a scenario where cash is depleted at an ATM for a duration, and no withdrawals are possible during that duration.
    The depletion occurs in the prediction horizon, and should be deductable by a model from the text context.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56

        # Select a dataset
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
            history_factor=self.random.randint(2, 4),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        if dataset_name == "nn5_daily_without_missing":
            drop_duration = self.random.choice(
                list(range(1, 12))
            )  # Arbitrarily picked from 1-12 days
            future_series.index = future_series.index.to_timestamp()
            drop_start_date = self.random.choice(
                future_series.index[
                    :-12
                ]  # Starting point is anywhere from start of series to max(drop_duration) + 1 points before the series. +1 is so as to not have the drop not completely at the end of the pred.
            )
            # Introduce a zero period in the future series at that duration
            drop_start_point = future_series.index.get_loc(drop_start_date)
            future_series.iloc[drop_start_point : drop_start_point + drop_duration] = 0

            # Convert history index to timestamp for consistency
            history_series.index = history_series.index.to_timestamp()

        # Transform
        history_series = add_realistic_noise(history_series, self.random)
        future_series = add_realistic_noise(future_series, self.random)

        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England."
        scenario = self.get_scenario(drop_start_date, drop_duration)

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

    def get_scenario(self, drop_start_date, drop_duration):
        scenario = f"Consider that cash is depleted in the ATM from {datetime_to_str(drop_start_date)}, for {drop_duration} {'day' if drop_duration == 1 else 'days'}, resulting in no withdrawals during that period."  # TODO: May also specify drop end date instead of the drop duration.
        return scenario


class ATMBuildingClosedTask(CashDepletedinATMScenarioTask):
    """
    Same as CashDepletedinATMScenarioTask, except that the context says the building is closed, and the model needs to use deduction to solve the task
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_cov"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, drop_start_date, drop_duration):
        scenario = f"Consider that the building which contains the ATM is closed from {datetime_to_str(drop_start_date)}, for {drop_duration} {'day' if drop_duration == 1 else 'days'}."  # TODO: May also specify drop end date instead of the drop duration.
        return scenario


class ATMUnderPeriodicMaintenanceTaskWithConclusion(UnivariateCRPSTask):
    """
    This task considers that an ATM is under periodic maintenance in the history repeatedly at certain intervals, which leads to misleading history. The context provides background information about this period.
    This period should be ignored by the forecasting algorithm in its forecasts.
    """

    # XXX: No c_h since the context doesn't say what happened due to maintenance
    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_cov"]
    _skills = UnivariateCRPSTask._skills + ["instruction following", "reasoning: math"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56

        # Select a dataset
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

        history_factor = self.random.randint(2, 4)
        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=history_factor,
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        history_series.index = history_series.index.to_timestamp()

        drop_duration = self.random.choice(
            list(range(4, 8))
        )  # Arbitrarily picked from 2-8 days
        drop_spacing = self.random.choice(
            list(range(6, 18))
        )  # Spacing between each drop would be randomly picked from 6 to 18 days
        drop_start_date = self.random.choice(
            history_series.index[
                :-56
            ]  # Starting point is anywhere from start of series to 56 days before the end. This is done so we can get multiple such drops in the series going forward in the history. (56 since that's the prediction length of this dataset)
        )
        drop_start_point = history_series.index.get_loc(drop_start_date)
        start_point = drop_start_point
        # Add drops to the data
        while start_point + drop_duration < len(history_series.index):
            history_series.iloc[start_point : start_point + drop_duration] = 0
            start_point += drop_spacing
        # Convert future index to timestamp for consistency
        future_series.index = future_series.index.to_timestamp()

        # Transform
        history_series = add_realistic_noise(history_series, self.random)
        future_series = add_realistic_noise(future_series, self.random)

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = self.get_background(
            drop_duration, drop_spacing, drop_start_date
        )
        self.scenario = None

        # ROI parameters to add focus to the times where there would have been maintenance in the prediction region
        maintenance_hours_in_pred = []
        pred_start_point = start_point - len(history_series.index)
        while pred_start_point + drop_duration < len(future_series.index):
            # Starting point can be part of the history; we should only consider starting point from prediction
            # But we don't want to modify pred_start_point, so creating a new variable pred_start_point_modified
            # pred_start_point + drop_duration will definitely be part of the prediction horizon
            if pred_start_point < 0:
                pred_start_point_modified = 0
            else:
                pred_start_point_modified = pred_start_point
            maintenance_hours_in_pred.extend(
                list(range(pred_start_point_modified, pred_start_point + drop_duration))
            )
            pred_start_point += drop_spacing
        self.region_of_interest = maintenance_hours_in_pred

    def get_background(self, drop_duration, drop_spacing, drop_start_date):
        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England."  # This is generic background information common to all NN5 tasks
        background += f" The ATM was under maintenance for {drop_duration} {'day' if drop_duration == 1 else 'days'}, periodically every {drop_spacing} days, starting from {datetime_to_str(drop_start_date)}, resulting in no withdrawals recorded. Assume that the ATM will not be in maintenance in the future."
        return background


class ATMUnderPeriodicMaintenanceTaskWithConclusionLessExplicit(
    ATMUnderPeriodicMaintenanceTaskWithConclusion
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_cov"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "reasoning: deduction",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_background(self, drop_duration, drop_spacing, drop_start_date):
        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England."  # This is generic background information common to all NN5 tasks
        background += f" The ATM was under maintenance for various periods, resulting in no withdrawals recorded. Assume that the ATM will not be in maintenance in the future."
        return background


class ATMUnderPeriodicMaintenanceTaskWithoutConclusion(
    ATMUnderPeriodicMaintenanceTaskWithConclusion
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_cov"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "reasoning: deduction",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_background(self, drop_duration, drop_spacing, drop_start_date):
        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England."  # This is generic background information common to all NN5 tasks
        background += f" The ATM was under maintenance for {drop_duration} {'day' if drop_duration == 1 else 'days'}, periodically every {drop_spacing} days, starting from {datetime_to_str(drop_start_date)}. Assume that the ATM will not be in maintenance in the future."
        return background


class ATMUnderPeriodicMaintenanceWithRandomValuesTask(UnivariateCRPSTask):
    """
    This task considers that an ATM is under periodic maintenance in the history repeatedly at certain intervals, which leads to misleading history. The context provides background information about this period.
    This period should be ignored by the forecasting algorithm in its forecasts.

    NOTE: This task is too hard right now since the randomly sampled data looks too similar to the other data. So it's hard even for humans. Hence, not putting it in __TASKS__ for now.

    """

    __version__ = "0.0.2"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56 (~~2 days)

        # Select a dataset
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

        history_factor = self.random.randint(2, 4)
        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=history_factor,
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        history_series.index = history_series.index.to_timestamp()

        drop_duration = self.random.choice(
            list(range(2, 7))
        )  # Arbitrarily picked from 2-6 days
        drop_spacing = self.random.choice(
            list(range(6, 24))
        )  # Spacing between each drop would be randomly picked from 6 to 23 days
        drop_start_date = self.random.choice(
            history_series.index[
                :-24
            ]  # Starting point is anywhere from start of series to 24 days before the end (arbitrary). This is done so we can get multiple such drops in the series going forward in the history.
        )
        drop_start_point = history_series.index.get_loc(drop_start_date)
        start_point = drop_start_point
        # Calculate the probability distribution of the original array
        unique_values, _ = np.unique(history_series.to_numpy(), return_counts=True)
        # Add drops to the data
        while start_point + drop_duration < len(history_series.index):
            history_series.iloc[start_point : start_point + drop_duration] = (
                self.random.choice(unique_values, size=drop_duration)
            )
            start_point += drop_spacing
        # Convert future index to timestamp for consistency
        future_series.index = future_series.index.to_timestamp()

        # Transform
        history_series = add_realistic_noise(history_series, self.random)
        future_series = add_realistic_noise(future_series, self.random)

        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England."  # This is generic background information common to all NN5 tasks
        background += f" The ATM was under maintenance for {drop_duration} {'day' if drop_duration == 1 else 'days'}, periodically every {drop_spacing} days, starting from {datetime_to_str(drop_start_date)}, resulting in corrupted values in the data. Assume that the ATM will not be in maintenance in the future."

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = None


class IncreasedWithdrawalScenario(UnivariateCRPSTask):
    """
    This task considers a scenario where the number of ATM withdrawals possible by a person are increased by the bank, resulting in more than usual withdrawals.
    This is a scenario that occurs in the prediction horizon, and should be deductable by a model from the text context.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_i", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def random_instance(self):
        datasets = [
            "nn5_daily_without_missing"
        ]  # nn5_daily_without_missing has a prediction length of 56

        # Select a dataset
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
            history_factor=self.random.randint(2, 4),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        limit_off_duration = self.random.choice(
            list(range(4, 12))
        )  # Arbitrarily picked from 4-11 days
        future_series.index = future_series.index.to_timestamp()
        limit_off_start_date = self.random.choice(
            future_series.index[
                :-12
            ]  # Starting point is anywhere from start of series to max(drop_duration) + 1 points before the series. +1 is so as to have the drop not completely at the end of the pred.
        )  # Arbitrary start point for now
        # Introduce a zero period in the future series at a particular duration
        limit_off_start_point = future_series.index.get_loc(limit_off_start_date)
        # Determine increase in number of withdrawals
        increase_factor = self.random.choice(
            list(range(3, 6))
        )  # Randomly picked from 3 to 5. Arbitrary.
        future_series.iloc[
            limit_off_start_point : limit_off_start_point + limit_off_duration
        ] = (
            increase_factor
            * future_series.iloc[
                limit_off_start_point : limit_off_start_point + limit_off_duration
            ]
        )

        # Convert history index to timestamp for consistency
        history_series.index = history_series.index.to_timestamp()

        # Transform
        history_series = add_realistic_noise(history_series, self.random)
        future_series = add_realistic_noise(future_series, self.random)

        event_type = self.random.choice(
            ["festival", "holiday", "celebration", "party", "music concert", "carnival"]
        )
        background = f"This is the number of cash withdrawals from an automated teller machine (ATM) in an arbitrary location in England."
        # TODO: Could be modified to involve deduction by saying "leading to x many more people in the area" --> should believe x times number of withdrawals
        scenario = f"Suppose that there is a {event_type} from {datetime_to_str(limit_off_start_date)}, for {limit_off_duration} {'day' if limit_off_duration == 1 else 'days'} leading to more people in the area, and {increase_factor} times the number of usual withdrawals during that period."
        # Covariate task: Event or not

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = scenario

        # ROI metric parameters
        self.region_of_interest = slice(
            limit_off_start_point, limit_off_start_point + limit_off_duration
        )


__TASKS__ = [
    CashDepletedinATMScenarioTask,
    ATMBuildingClosedTask,
    ATMUnderPeriodicMaintenanceTaskWithConclusion,
    ATMUnderPeriodicMaintenanceTaskWithConclusionLessExplicit,
    ATMUnderPeriodicMaintenanceTaskWithoutConclusion,
    IncreasedWithdrawalScenario,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            ATMBuildingClosedTask,
        ],
    ),
    WeightCluster(
        weight=1,
        tasks=[
            CashDepletedinATMScenarioTask,
            IncreasedWithdrawalScenario,
        ],
    ),
    WeightCluster(
        weight=1,
        tasks=[
            ATMUnderPeriodicMaintenanceTaskWithConclusion,
            ATMUnderPeriodicMaintenanceTaskWithConclusionLessExplicit,
            ATMUnderPeriodicMaintenanceTaskWithoutConclusion,
        ],
    ),
]
