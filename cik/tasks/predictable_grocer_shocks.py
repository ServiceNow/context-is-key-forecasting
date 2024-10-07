import json
import os
import pandas as pd
from pandas.tseries.offsets import WeekOfMonth

from ..base import UnivariateCRPSTask
from ..data.dominicks import (
    download_dominicks,
    DOMINICK_JSON_PATH,
    DOMINICK_CSV_PATH,
)
from ..utils import get_random_window_univar
from . import WeightCluster


class PredictableGrocerPersistentShockUnivariateTask(UnivariateCRPSTask):
    """
    A task where the time series contains spikes that are predictable based on the
    contextual information provided with the data. The spikes should be reflected in
    the forecast.
    Note: this does NOT use the Monash dominick's dataset, which is transformed with no
    meaningful context.
    Context: synthetic (GPT-generated then edited)
    Series: modified
    Dataset: Dominick's grocer dataset (daily)
    Parameters:
    -----------
    fixed_config: dict
        A dictionary containing fixed parameters for the task
    seed: int
        Seed for the random number generator
    GROCER_SALES_INFLUENCES_PATH: str
        Path to the JSON file containing the sales influences.
    DOMINICK_GROCER_SALES_PATH: str
        Path to the filtered Dominick's grocer dataset.
        Filtered for a subset of products for which we generated influences.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):
        self.init_data()
        self.dominick_grocer_sales_path = DOMINICK_CSV_PATH
        self.grocer_sales_influences_path = DOMINICK_JSON_PATH
        with open(self.grocer_sales_influences_path, "r") as file:
            self.influences = json.load(file)

        super().__init__(seed=seed, fixed_config=fixed_config)

    def init_data(self):
        """
        Check integrity of data files and download if needed.

        """
        if not os.path.exists(DOMINICK_JSON_PATH):
            raise FileNotFoundError("Missing Dominick json file.")
        if not os.path.exists(DOMINICK_CSV_PATH):
            download_dominicks()

    def random_instance(self):
        dataset = pd.read_csv(self.dominick_grocer_sales_path)
        dataset["date"] = pd.to_datetime(dataset["datetime"])
        dataset = dataset.set_index("date")

        stores = dataset["store"].unique()

        self.prediction_length = self.random.randint(7, 30)

        for counter in range(100000):
            # pick a random sales category and store
            sales_category = self.random.choice(self.sales_categories)
            store = self.random.choice(stores)

            # select a random series
            series = dataset[dataset["store"] == store][sales_category]

            if (series == 0).mean() > 0.5:
                continue

            # select a random window
            history_factor = self.random.randint(3, 7)
            if len(series) > (history_factor + 1) * self.prediction_length:
                window = get_random_window_univar(
                    series,
                    prediction_length=self.prediction_length,
                    history_factor=history_factor,
                    random=self.random,
                )
                break  # Found a valid window, stop the loop
        else:
            raise ValueError("Could not find a valid window.")

        covariates = self.get_covariates(dataset, store, sales_category, window)
        window = self.mitigate_memorization(window)

        # extract the history and future series
        history_series = window.iloc[: -self.prediction_length]
        future_series = window.iloc[-self.prediction_length :]
        ground_truth = future_series.copy()

        # choose an influence and a relative impact from the influence
        shock_delay_in_days = self.random.randint(0, self.prediction_length - 1)
        shock_duration = self.get_shock_duration(shock_delay_in_days)

        direction = self.random.choice(["positive", "negative"])
        size = self.random.choice(["small", "medium", "large"])
        influence_info = self.influences[sales_category][direction][size]
        impact_range = influence_info["impact"]
        self.min_magnitude, self.max_magnitude = map(
            lambda x: int(x.strip("%")), impact_range.split("-")
        )
        impact_magnitude = self.random.randint(self.min_magnitude, self.max_magnitude)

        # apply the influence to the future series
        future_series[shock_delay_in_days : shock_delay_in_days + shock_duration] = (
            self.apply_influence_to_series(
                future_series[
                    shock_delay_in_days : shock_delay_in_days + shock_duration
                ],
                impact_magnitude,
                direction,
            )
        )
        if covariates is not None:
            covariates = self.mitigate_memorization(covariates)

        self.min_magnitude = self.min_magnitude
        self.max_magnitude = self.max_magnitude
        self.impact_magnitude = impact_magnitude
        self.direction = direction
        self.ground_truth = ground_truth

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = self.get_background_context(sales_category)
        self.scenario = self.get_scenario_context(
            shock_delay_in_days, influence_info, covariates
        )

        self.region_of_interest = slice(
            shock_delay_in_days, shock_delay_in_days + shock_duration
        )

    def get_shock_duration(self, shock_delay_in_days):
        """
        Shock is persistent for the rest of the prediction length,
        """
        return self.prediction_length - shock_delay_in_days + 1

    def mitigate_memorization(self, window):
        """
        Mitigate memorization by doubling the values of the series and updating the year of the timesteps.
        """
        window = window.copy()
        window *= 2

        # Get the start month, week, and day of the week from the original window
        start_date = window.index[0]
        month = start_date.month
        day_of_week = start_date.weekday()  # Monday=0, Sunday=6

        # Find the nth occurrence of the weekday within the month
        week_of_month = (start_date.day - 1) // 7  # Get the week number (0-based)

        if week_of_month == 4:
            week_of_month = 0
            month += 1
            if month > 12:
                month = 1

        # Pandas occasionally moves the date over to the next month, so we go back one day
        first_of_month_2024_minus_1 = pd.Timestamp(2024, month, 1) - pd.DateOffset(
            days=1
        )
        target_start_date = first_of_month_2024_minus_1 + WeekOfMonth(
            week=week_of_month, weekday=day_of_week
        )

        # Generate new index starting from target_start_date
        new_index = pd.date_range(
            start=target_start_date, periods=len(window), freq="D"
        )

        # Assign the newly generated index to the window
        window.index = new_index

        return window

    def get_shock_description(self, shock_delay_in_days, influence_info):
        return influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )

    def apply_influence_to_series(self, series, relative_impact, direction):
        """
        Apply a relative impact to a series
        """
        if direction == "positive":
            series += series * (relative_impact / 100)
        else:
            series -= series * (relative_impact / 100)

        return series

    def get_background_context(self, sales_category):
        """
        Get the background context of the event.

        """
        return f"The following series contains {sales_category.capitalize()} sales in dollars of a grocery store."

    def get_covariates(self, dataset, store, sales_category, window):
        """
        Get the covariates of the event.

        """
        return None

    def get_scenario_context(self, shock_delay_in_days, influence_info, covariates):
        """
        Get the context of the event.

        """

        shock_description = influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )
        shock_description = shock_description.replace(
            "{impact}", str(self.impact_magnitude) + "%"
        )
        shock_description += f" This impact is expected to last for at least {self.prediction_length - shock_delay_in_days} days."
        return shock_description

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return 7


class PredictableGrocerTemporaryShockUnivariateTask(
    PredictableGrocerPersistentShockUnivariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def get_shock_duration(self, shock_delay_in_days):
        """
        Shock is temporary for a random duration between of at least 2 days.
        """
        return self.random.randint(2, self.prediction_length - shock_delay_in_days + 1)


class PredictableGrocerPersistentShockUnivariateBeerTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["beer"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateMeatTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["meat"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateGroceryTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["grocery"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateDairyTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["dairy"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateProduceTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["produce"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateFrozenTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["frozen"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariatePharmacyTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["pharmacy"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateBakeryTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["bakery"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockUnivariateGmTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["gm"]
        super().__init__(seed=seed, fixed_config=fixed_config)

    def get_background_context(self, sales_category):
        """
        Get the background context of the event.

        """
        return "The following series contains general merchandise sales in dollars of a grocery store."


class PredictableGrocerPersistentShockUnivariateFishTask(
    PredictableGrocerPersistentShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["fish"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateBeerTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["beer"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateMeatTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["meat"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateGroceryTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["grocery"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateDairyTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["dairy"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateProduceTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["produce"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateFrozenTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["frozen"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariatePharmacyTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["pharmacy"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateBakeryTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["bakery"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockUnivariateGmTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["gm"]
        super().__init__(seed=seed, fixed_config=fixed_config)

    def get_background_context(self, sales_category):
        """
        Get the background context of the event.

        """
        return "The following series contains general merchandise sales in dollars of a grocery store."


class PredictableGrocerTemporaryShockUnivariateFishTask(
    PredictableGrocerTemporaryShockUnivariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["fish"]
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateTask(
    PredictableGrocerPersistentShockUnivariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):
        super().__init__(seed=seed, fixed_config=fixed_config)

    def get_covariates(self, dataset, store, sales_category, window):
        """
        Get the covariates of the event.

        """
        cov_series = dataset[dataset["store"] == store][self.chosen_covariate]
        cov_series = cov_series.loc[window.index]
        return cov_series

    def get_scenario_context(self, shock_delay_in_days, influence_info, covariates):
        """
        Get the context of the event.

        """
        possible_covariates = {
            "custcoun": "the number of customers in the store",
        }

        shock_description = influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )
        shock_description = shock_description.replace(
            "{impact}", str(self.impact_magnitude) + "%"
        )
        shock_description += f" This impact is expected to last for at least {self.prediction_length - shock_delay_in_days} days."

        shock_description += f" The following series contains {possible_covariates[self.chosen_covariate]} in the store: \n{covariates.to_string(index=True)}"

        return shock_description


class PredictableGrocerPersistentShockCovariateBeerTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["beer"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateMeatTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["meat"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateGroceryTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["grocery"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateDairyTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["dairy"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateProduceTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["produce"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateFrozenTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["frozen"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariatePharmacyTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["pharmacy"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateBakeryTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["bakery"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerPersistentShockCovariateGmTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["gm"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)

    def get_background_context(self, sales_category):
        """
        Get the background context of the event.

        """
        return "The following series contains general merchandise sales in dollars of a grocery store."


class PredictableGrocerPersistentShockCovariateFishTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["fish"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateBeerTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["beer"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateMeatTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["meat"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateGroceryTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["grocery"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateDairyTask(
    PredictableGrocerPersistentShockCovariateTask
):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["dairy"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateProduceTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["produce"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateFrozenTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["frozen"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariatePharmacyTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["pharmacy"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateBakeryTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["bakery"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


class PredictableGrocerTemporaryShockCovariateGmTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["gm"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)

    def get_background_context(self, sales_category):
        """
        Get the background context of the event.

        """
        return "The following series contains general merchandise sales in dollars of a grocery store."


class PredictableGrocerTemporaryShockCovariateFishTask(
    PredictableGrocerPersistentShockCovariateTask
):

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["reasoning: deduction"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
    ):

        self.sales_categories = ["fish"]
        self.chosen_covariate = "custcoun"
        super().__init__(seed=seed, fixed_config=fixed_config)


__TASKS__ = [
    PredictableGrocerPersistentShockUnivariateBeerTask,
    PredictableGrocerPersistentShockUnivariateMeatTask,
    PredictableGrocerPersistentShockUnivariateGroceryTask,
    PredictableGrocerPersistentShockUnivariateDairyTask,
    PredictableGrocerPersistentShockUnivariateProduceTask,
    PredictableGrocerPersistentShockUnivariateFrozenTask,
    PredictableGrocerPersistentShockUnivariatePharmacyTask,
    PredictableGrocerPersistentShockUnivariateBakeryTask,
    PredictableGrocerPersistentShockUnivariateGmTask,
    PredictableGrocerPersistentShockUnivariateFishTask,
    PredictableGrocerTemporaryShockUnivariateBeerTask,
    PredictableGrocerTemporaryShockUnivariateMeatTask,
    PredictableGrocerTemporaryShockUnivariateGroceryTask,
    PredictableGrocerTemporaryShockUnivariateDairyTask,
    PredictableGrocerTemporaryShockUnivariateProduceTask,
    PredictableGrocerTemporaryShockUnivariateFrozenTask,
    PredictableGrocerTemporaryShockUnivariatePharmacyTask,
    PredictableGrocerTemporaryShockUnivariateBakeryTask,
    PredictableGrocerTemporaryShockUnivariateGmTask,
    PredictableGrocerTemporaryShockUnivariateFishTask,
    PredictableGrocerPersistentShockCovariateBeerTask,
    PredictableGrocerPersistentShockCovariateMeatTask,
    PredictableGrocerPersistentShockCovariateGroceryTask,
    PredictableGrocerPersistentShockCovariateDairyTask,
    PredictableGrocerPersistentShockCovariateProduceTask,
    PredictableGrocerPersistentShockCovariateFrozenTask,
    PredictableGrocerPersistentShockCovariatePharmacyTask,
    PredictableGrocerPersistentShockCovariateBakeryTask,
    PredictableGrocerPersistentShockCovariateGmTask,
    PredictableGrocerPersistentShockCovariateFishTask,
    PredictableGrocerTemporaryShockCovariateBeerTask,
    PredictableGrocerTemporaryShockCovariateMeatTask,
    PredictableGrocerTemporaryShockCovariateGroceryTask,
    PredictableGrocerTemporaryShockCovariateDairyTask,
    PredictableGrocerTemporaryShockCovariateProduceTask,
    PredictableGrocerTemporaryShockCovariateFrozenTask,
    PredictableGrocerTemporaryShockCovariatePharmacyTask,
    PredictableGrocerTemporaryShockCovariateBakeryTask,
    PredictableGrocerTemporaryShockCovariateGmTask,
    PredictableGrocerTemporaryShockCovariateFishTask,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=__TASKS__,
    ),
]
