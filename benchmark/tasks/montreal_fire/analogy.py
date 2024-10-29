"""
Tasks for the Montreal Fire dataset.

"""

from .utils import calculate_yearly_sum_stats_for_months
from ...base import UnivariateCRPSTask
from ...data.montreal_fire.load import (
    get_incident_log,
    get_time_count_series_by_borough,
)
from .. import WeightCluster


class MontrealFireNauticalRescueAnalogyTask(UnivariateCRPSTask):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_i"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: analogy", "retrieval: context"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def __init__(
        self,
        seed=None,
        fixed_config=None,
        explicit_location=True,
        include_reference_location=True,
        with_water=None,
    ):
        """
        Montreal Nautical Rescue Prediction Task. This tests analogical reasoning and retrieval.

        Parameters:
        -----------
        seed: int
            Seed for the random number generator
        fixed_config: dict
            Fixed configuration for the task
        explicit_location: bool
            Whether to explicitly mention the location (near water or not) of the target borough in the context
        include_reference_location: bool
            Whether to specify the reference borough locations (near water or not) in the context
        with_water: bool (default: None)
            If set, whether the target borough should be near water (True) or not (False).
            Otherwise, it will be picked at random.

        """
        # Task configuration
        self.explicit_location = explicit_location
        self.include_reference_location = include_reference_location
        self.with_water = with_water
        self.history_length = 3  # Months
        self.prediction_length = 6  # Months

        self.all_series = self._get_data()

        super().__init__(seed=seed, fixed_config=fixed_config)

    @property
    def seasonal_period(self):
        return -1

    def _get_data(self):
        incidents = get_incident_log()
        return get_time_count_series_by_borough(
            incidents, "Nautical rescue", frequency="M"
        )

    def random_instance(self):
        fake_water = [
            "Côte-des-Rivières",
            "Montréal-sur-le-Fleuve",
            "Vieux-Port Est",
            "Baie-des-Marins",
        ]
        fake_nowater = [
            "Montagne-Centre",
            "Bois-des-Écureuils",
            "Centre-Ville Est",
            "Sommet-de-la-Montagne",
        ]

        # Populations (taken from wikipedia on August 27, 2024)
        boroughs = {
            "Saint-Laurent": {"pop": 98828, "water": False},
            "Le Plateau-Mont-Royal": {"pop": 104000, "water": False},
            "Rivière-des-Prairies-Pointe-aux-Trembles": {"pop": 107941, "water": True},
            "Ahuntsic-Cartierville": {"pop": 134245, "water": True},
            "Verdun": {"pop": 69229, "water": True},
        }

        if self.with_water is None:
            with_water = self.random.choice([False, True])
        else:
            with_water = self.with_water

        # Target borough
        # ... give it a name that suggest's if it's near water or not
        fake_name = self.random.choice(fake_water if with_water else fake_nowater)
        # ... select a borough with the desired water property to serve as target series
        target = self.random.choice(
            [k for k, v in boroughs.items() if v["water"] == with_water]
        )

        # Context: create the story
        context = f"{self.history_length} month{'s' if self.history_length > 1 else ''} ago, the city of Montreal inaugurated a new borough called {fake_name} (pop.: {boroughs[target]['pop']}). "
        if self.explicit_location:
            if with_water:
                context += "This borough is adjacent to the Saint-Lawrence River. "
            else:
                context += "This borough is landlocked in the heart of the city. "
        context += f"\nThe Fire Chief aims to strategically allocate resources and seeks an estimate of the anticipated number of nautical rescues for the next {self.prediction_length} months.\n"

        # Context: add historical information about two reference boroughs (used for analogy)
        # ... configuration for past year statistics
        stat_months = [5, 6, 7, 8, 9]
        stats_month_start_str = "May"
        stats_month_end_str = "September"
        stats_cutoff_year = 2024
        # ... with water
        ref = self.random.choice(
            [k for k, v in boroughs.items() if k != target and v["water"]]
        )
        desc = ", major bodies of water" if self.include_reference_location else ""
        count_stats = calculate_yearly_sum_stats_for_months(
            self.all_series[ref], months=stat_months, cutoff_year=stats_cutoff_year
        )
        context += f"\nFor reference, here are some statistics for two of the city's boroughs.\nThe values are historical yearly incident counts [min, max] between {stats_month_start_str} and {stats_month_end_str} for the past {count_stats['n']} years:\n"
        context += f"* {ref}{desc} (pop.: {boroughs[ref]['pop']}): [{count_stats['min']}, {count_stats['max']}] incidents\n"
        # ... without water
        ref = self.random.choice(
            [k for k, v in boroughs.items() if k != target and not v["water"]]
        )
        desc = ", no major bodies of water" if self.include_reference_location else ""
        count_stats = calculate_yearly_sum_stats_for_months(
            self.all_series[ref], months=stat_months, cutoff_year=stats_cutoff_year
        )
        context += f"* {ref}{desc} (pop.: {boroughs[ref]['pop']}): [{count_stats['min']}, {count_stats['max']}] incidents\n"

        # Create the instance's time series
        target_year = 2024
        target_months = [1, 2, 3]

        target_series = self.all_series[target]
        forecast_start_index = self.random.choice(
            target_series.loc[
                (target_series.index.year == target_year)
                & (target_series.index.month.isin(target_months))
            ].index
        )
        window = target_series[
            forecast_start_index
            - self.history_length : forecast_start_index
            + self.prediction_length
            - 1
        ]
        self.past_time = window[: forecast_start_index - 1].to_frame()
        self.future_time = window[forecast_start_index:].to_frame()
        assert len(self.past_time) == self.history_length
        assert len(self.future_time) == self.prediction_length

        self.background = context


class MontrealFireNauticalRescueAnalogyFullLocalizationNoWaterTask(
    MontrealFireNauticalRescueAnalogyTask
):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            explicit_location=True,
            include_reference_location=True,
            with_water=False,
        )


class MontrealFireNauticalRescueAnalogyFullLocalizationWaterTask(
    MontrealFireNauticalRescueAnalogyTask
):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            explicit_location=True,
            include_reference_location=True,
            with_water=True,
        )


class MontrealFireNauticalRescueAnalogyFullLocalizationMaybeWaterTask(
    MontrealFireNauticalRescueAnalogyTask
):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            explicit_location=True,
            include_reference_location=True,
            with_water=None,
        )


class MontrealFireNauticalRescueAnalogyTargetLocalizationNoWaterTask(
    MontrealFireNauticalRescueAnalogyTask
):

    _skills = MontrealFireNauticalRescueAnalogyTask._skills + ["retrieval: memory"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            explicit_location=True,
            include_reference_location=False,
            with_water=False,
        )


class MontrealFireNauticalRescueAnalogyTargetLocalizationWaterTask(
    MontrealFireNauticalRescueAnalogyTask
):

    _skills = MontrealFireNauticalRescueAnalogyTask._skills + ["retrieval: memory"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            explicit_location=True,
            include_reference_location=False,
            with_water=True,
        )


class MontrealFireNauticalRescueAnalogyTargetLocalizationMaybeWaterTask(
    MontrealFireNauticalRescueAnalogyTask
):

    _skills = MontrealFireNauticalRescueAnalogyTask._skills + ["retrieval: memory"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed=None, fixed_config=None):
        super().__init__(
            seed=seed,
            fixed_config=fixed_config,
            explicit_location=True,
            include_reference_location=False,
            with_water=None,
        )


# XXX: Commented out since this requires guessing our own opinion about if a
#      borough name sounds aquatic or not.
# class MontrealFireNauticalRescueAnalogyReferenceLocalizationTask(
#     MontrealFireNauticalRescueAnalogyTask
# ):
#     _skills = MontrealFireNauticalRescueAnalogyTask._skills + ["reasoning: deduction"]
#     __version__ = "0.0.1"  # Modification will trigger re-caching

#     def __init__(self, seed=None, fixed_config=None):
#         super().__init__(
#             seed=seed,
#             fixed_config=fixed_config,
#             explicit_location=False,
#             include_reference_location=True,
#         )
#
#
# class MontrealFireNauticalRescueAnalogyNoLocalizationTask(
#     MontrealFireNauticalRescueAnalogyTask
# ):
#     _skills = MontrealFireNauticalRescueAnalogyTask._skills + ["reasoning: deduction"]
#     __version__ = "0.0.1"  # Modification will trigger re-caching

#     def __init__(self, seed=None, fixed_config=None):
#         super().__init__(
#             seed=seed,
#             fixed_config=fixed_config,
#             explicit_location=False,
#             include_reference_location=False,
#         )


__TASKS__ = [
    # Skip these since, taken by themselves, the NoWater tasks have nearly zero ground-truth forecast.
    # This leads to a very aggressive normalization of their CRPS scores, heavily penalizing any error.
    # MontrealFireNauticalRescueAnalogyFullLocalizationNoWaterTask,
    # MontrealFireNauticalRescueAnalogyTargetLocalizationNoWaterTask,
    # MontrealFireNauticalRescueAnalogyFullLocalizationWaterTask,
    # MontrealFireNauticalRescueAnalogyTargetLocalizationWaterTask,
    MontrealFireNauticalRescueAnalogyFullLocalizationMaybeWaterTask,
    MontrealFireNauticalRescueAnalogyTargetLocalizationMaybeWaterTask,
    # XXX: see above comment for reason of leaving out
    # MontrealFireNauticalRescueAnalogyReferenceLocalizationTask,
    # MontrealFireNauticalRescueAnalogyNoLocalizationTask,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            MontrealFireNauticalRescueAnalogyFullLocalizationMaybeWaterTask,
            MontrealFireNauticalRescueAnalogyTargetLocalizationMaybeWaterTask,
        ],
    )
]
