from ...data.montreal_fire.load import (
    get_incident_log,
    get_time_count_series_by_borough,
)
from .short_history import MontrealFireShortHistoryTask
from .utils import calculate_yearly_sum_stats_for_months


class MontrealFireCausalConfoundingTask(MontrealFireShortHistoryTask):
    """
    The task mentions an intervention performed on a confounder.
    If the model uses the context but does not understand causality,
    it will forecast that the incidents quickly go to zero. To ensure
    that the model fails if they don't use the context, we also make
    this a short history task. If the model simply disregards the context,
    it will forecast an upward slope and fail.

    Parameters:
    -----------
    target_series: str
        The series to forecast
    covariate_series: str
        The covariate on which the intervention will be performed
    action_descriptions: list of str
        The a description of every possible intervention
    seed : int, optional
        The random seed to use.
    fixed_config : dict, optional
        A dictionary of fixed configuration parameters.
    history_start_month : int, default 12
        The month in which the historical data starts.
    history_length : int, default 6
        The number of months of historical data to use.
    min_occurrences: int, default 10
        The minimum number yearly occurrences of a "series" incidents each year (on average)
        in order for a borough's data to be included in the task instances.

    """

    _context_sources = MontrealFireShortHistoryTask._context_sources + [
        "c_cov",
        "c_causal",
        "c_f",
    ]
    _skills = MontrealFireShortHistoryTask._skills + ["reasoning: causal"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        target_series,
        covariate_series,
        action_descriptions,
        seed: int = None,
        fixed_config: dict | None = None,
        history_start_month=12,
        history_length=6,
        min_occurrences=10,
    ):

        self.target_series = target_series
        self.covariate_series = covariate_series
        self.action_descriptions = action_descriptions

        assert (
            history_length < 12
        ), "History length must be less than 12 months. To leave room for forecasting."

        super().__init__(
            seed,
            fixed_config,
            series=target_series,
            history_start_month=history_start_month,
            history_length=history_length,
            include_max_month=True,
            min_occurrences=min_occurrences,
        )

    def random_instance(self):
        super().random_instance()

        action_desc = self.random.choice(self.action_descriptions)
        target_series = self.target_series.lower()
        covariate_series = self.covariate_series.lower()
        self.background += f"""

The Mayor is determined to completely eradicate this kind of incident.
Fortunately, the city's public safety research group, a team of highly qualified experts, identified that {target_series}s and {covariate_series}s tend to co-occur.
When the amount of {target_series}s increases, the amount of {covariate_series}s also tends to increase. The same applies to decreases.

The Mayor has a plan: they will implement {action_desc} starting on {self.forecast_start_date}.
In a recent interview, they claimed, "This is a bulletproof plan, and I am certain it will immediately put an end to {target_series}s."
"""


class MontrealFireFieldAndTrashCausalConfoundingTask(MontrealFireCausalConfoundingTask):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: dict | None = None):
        super().__init__(
            target_series="Field fire",
            covariate_series="Trash fire",
            action_descriptions=[
                "daily spraying of all piles of trash with water",
                "daily spraying of all piles of trash with fire retardant foam",
                "daily collection of all trash and storage in a fire-proof facility",
            ],
            seed=seed,
            fixed_config=fixed_config,
        )


class MontrealFireTrashAndNauticalCausalConfoundingTask(
    MontrealFireCausalConfoundingTask
):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: dict | None = None):
        super().__init__(
            target_series="Trash fire",
            covariate_series="Nautical rescue",
            action_descriptions=[
                "a strict prohibition of all nautical activities",
                "constrained zones with great supervision for nautical activities and astonishing fines for violators",
                "high metal fences that prevent access to all water basins in and around the city",
            ],
            seed=seed,
            fixed_config=fixed_config,
        )


class MontrealFireFieldAndGasCausalConfoundingTask(MontrealFireCausalConfoundingTask):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: dict | None = None):
        super().__init__(
            target_series="Field fire",
            covariate_series="Gas leak",
            action_descriptions=[
                "a strict prohibition of using any form of combustible gas in the city",
                "the immediate termination of gas sales and astonishing fines for anyone caught using some",
                "a new bylaw preventing the use of any combustible gas with severe penalties for offenders",
            ],
            seed=seed,
            fixed_config=fixed_config,
        )


class MontrealFireTrashAndNauticalCausalConfoundingTask(
    MontrealFireCausalConfoundingTask
):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: dict | None = None):
        super().__init__(
            target_series="Trash fire",
            covariate_series="Nautical rescue",
            action_descriptions=[
                "a strict prohibition of all nautical activities",
                "constrained zones with great supervision for nautical activities and astonishing fines for violators",
                "high metal fences that prevent access to all water basins in and around the city",
            ],
            seed=seed,
            fixed_config=fixed_config,
        )


class MontrealFireTrashAndBicycleCausalConfoundingTask(
    MontrealFireCausalConfoundingTask
):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: dict | None = None):
        super().__init__(
            target_series="Trash fire",
            covariate_series="Bicycle accident",
            action_descriptions=[
                "a strict prohibition of all cycling activities in the city",
                "the immediate imprisonment of anyone caught riding a bicycle",
            ],
            seed=seed,
            fixed_config=fixed_config,
        )


__TASKS__ = [
    MontrealFireFieldAndTrashCausalConfoundingTask,
    MontrealFireFieldAndGasCausalConfoundingTask,
    MontrealFireTrashAndNauticalCausalConfoundingTask,
    MontrealFireTrashAndBicycleCausalConfoundingTask,
]
