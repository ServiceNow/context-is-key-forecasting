import numpy as np

from calendar import month_name
from scipy import stats

from .utils import calculate_yearly_sum_stats_for_months
from ...base import UnivariateCRPSTask
from ...data.montreal_fire.load import (
    get_incident_log,
    get_time_count_series_by_borough,
)
from .. import WeightCluster


class MontrealFireShortHistoryTask(UnivariateCRPSTask):
    """
    A task that requires forecasting the number of incidents of a specific type in Montreal based on a short history of past incidents.

    Parameters:
    -----------
    seed : int, optional
        The random seed to use.
    fixed_config : dict, optional
        A dictionary of fixed configuration parameters.
    series : str, default 'Field fire'
        The type of incident to forecast.
    history_start_month : int, default 12
        The month in which the historical data starts.
    history_length : int, default 6
        The number of months of historical data to use.
    include_max_month : bool, default True
        Whether to include information about the month with the most incidents in the past.
    min_occurrences: int, default 10
        The minimum number yearly occurrences of a "series" incidents each year (on average)
        in order for a borough's data to be included in the task instances.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_i", "c_h"]
    _skills = UnivariateCRPSTask._skills + [
        "reasoning: math",
        "retrieval: memory",
        "reasoning: deduction",
    ]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(
        self,
        seed=None,
        fixed_config=None,
        series="Field fire",
        history_start_month=12,
        history_length=6,
        include_max_month=True,
        min_occurrences=10,
    ):
        self.series = series
        self.history_start_month = history_start_month
        self.history_length = history_length
        self.include_max_month = include_max_month
        self.min_occurrences = min_occurrences
        super().__init__(seed=seed, fixed_config=fixed_config)

        assert (
            history_length < 12
        ), "History length must be less than 12 months. To leave room for forecasting."

    @property
    def seasonal_period(self):
        return -1

    def _get_data(self):
        incidents = get_incident_log()

        all_series = get_time_count_series_by_borough(
            incidents, self.series, frequency="M"
        )

        valid_boroughs = []
        for borough, b_series in all_series.items():
            if borough == "Indéterminé":
                continue
            if (
                calculate_yearly_sum_stats_for_months(df=b_series, cutoff_year=2024)[
                    "mean"
                ]
                > self.min_occurrences
            ):
                valid_boroughs.append(borough)

        random_borough = self.random.choice(valid_boroughs)
        return all_series[random_borough], random_borough

    def random_instance(self):
        series, borough = self._get_data()

        valid_start_dates = [
            date
            for date in series.loc[series.index.month == self.history_start_month].index
            if len(series[date:]) >= 12
        ]

        self.history_start_date = self.random.choice(valid_start_dates)
        self.history_end_date = self.history_start_date + self.history_length - 1
        self.forecast_start_date = self.history_end_date + 1
        self.forecast_end_date = self.forecast_start_date + 12 - self.history_length

        self.past_time = series[
            self.history_start_date : self.history_end_date
        ].to_frame()
        self.future_time = series[
            self.forecast_start_date : self.forecast_end_date
        ].to_frame()

        self.background = "The Montreal Fire Department is in charge of responding to various kind of public safety incidents. "

        # Add randomness to the background
        if self.random.random() < 0.5:
            self.background += f"This series contains the number of {self.series.lower()} incidents responded to by the Montreal Fire Department"
        else:
            self.background += f"This is the number of {self.series.lower()} incidents responded to by Montreal firefighters"

        # Add borough information
        if borough.lower() == "total":
            self.background += "."
        else:
            self.background += f" in the borough of {borough}."

        # Get the other windows to calculate stats
        other = list(valid_start_dates)
        other.remove(self.history_start_date)

        # Calculate the total number of incidents
        count_per_year = calculate_yearly_sum_stats_for_months(
            series, list(range(1, 13)), cutoff_year=2024
        )["values"]
        del count_per_year[self.history_start_date.year]
        del count_per_year[self.forecast_end_date.year]

        # Add randomness to the background
        if self.random.random() < 0.5:
            self.background += f" In other years, the yearly average number of incidents was {np.mean(list(count_per_year.values())):.0f}"
        else:
            self.background += f" On average, they respond to {np.mean(list(count_per_year.values())):.0f} incidents per year"

        if self.include_max_month:
            max_month = stats.mode(
                [series[o : o + 12].idxmax().month for o in other]
            ).mode

            # Add randomness to the background
            if self.random.random() < 0.5:
                self.background += f" and the month with the most incidents was {month_name[max_month]}."
            else:
                self.background += (
                    f" with the busiest month being {month_name[max_month]}."
                )
        else:
            self.background += "."


class MontrealFireFieldFireExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Field fire",
            history_start_month=12,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireFieldFireImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Field fire",
            history_start_month=12,
            history_length=6,
            include_max_month=False,
        )


class MontrealFireTrashFireExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Trash fire",
            history_start_month=12,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireTrashFireImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Trash fire",
            history_start_month=12,
            history_length=6,
            include_max_month=False,
        )


class MontrealFireNauticalRescueExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Nautical rescue",
            history_start_month=12,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireNauticalRescueImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Nautical rescue",
            history_start_month=12,
            history_length=6,
            include_max_month=False,
        )


class MontrealFireIceRescueExplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Ice rescue",
            history_start_month=7,
            history_length=6,
            include_max_month=True,
        )


class MontrealFireIceRescueImplicitShortHistoryTask(MontrealFireShortHistoryTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            series="Ice rescue",
            history_start_month=7,
            history_length=6,
            include_max_month=False,
        )


__TASKS__ = [
    MontrealFireFieldFireExplicitShortHistoryTask,
    MontrealFireFieldFireImplicitShortHistoryTask,
    MontrealFireTrashFireExplicitShortHistoryTask,
    MontrealFireTrashFireImplicitShortHistoryTask,
    MontrealFireNauticalRescueExplicitShortHistoryTask,
    MontrealFireNauticalRescueImplicitShortHistoryTask,
    MontrealFireIceRescueExplicitShortHistoryTask,
    MontrealFireIceRescueImplicitShortHistoryTask,
]


__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            MontrealFireFieldFireExplicitShortHistoryTask,
            MontrealFireFieldFireImplicitShortHistoryTask,
            MontrealFireTrashFireExplicitShortHistoryTask,
            MontrealFireTrashFireImplicitShortHistoryTask,
            MontrealFireNauticalRescueExplicitShortHistoryTask,
            MontrealFireNauticalRescueImplicitShortHistoryTask,
            MontrealFireIceRescueExplicitShortHistoryTask,
            MontrealFireIceRescueImplicitShortHistoryTask,
        ],
    )
]
