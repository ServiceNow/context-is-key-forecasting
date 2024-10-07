from gluonts.dataset.util import to_pandas
from gluonts.time_feature import get_seasonality

from tactis.gluon.dataset import get_dataset

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str

from statsmodels.tsa.seasonal import STL

from abc import ABC, abstractmethod


class STLNoDescriptionContext:

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context():
        return None


class STLShortDescriptionContext:

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context():
        return """This task applies a multiplier to a component of the STL decomposition
        of the series."""


class STLMediumDescriptionContext:

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context():
        return """This task applies a multiplier to a component of the STL decomposition
        of the series. The seasonal-trend decomposition with LOESS (STL) is a method for
        decomposing a time series into trend, seasonal, and residual components."""


class STLLongDescriptionContext:

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context():
        return """This task applies a multiplier to a component of the STL decomposition
        of the series. The seasonal-trend decomposition with LOESS (STL) is a method for
        decomposing a time series into trend, seasonal, and residual components. The
        trend component represents the long-term progression of the series, the seasonal
        component represents the seasonal variation, and the residual component
        represents the noise in the series. """


class STLModifierTask(UnivariateCRPSTask):
    """
    A task where the series is first decomposed into trend, seasonality, and residuals
    using STL decomposition. One component is then modified and the series is recomposed.
    Possible variants include:
    - Multiplying the trend or seasonal component (in the pred or the hist)
    - Adding a constant value to the trend or seasonal component
    - Modifying the slope of the trend component
    - Messing with the frequency of the seasonal component (e.g. disregard warping)
    Time series: agnostic
    Context: synthetic
    Parameters:
    ----------
    target_component_name: str
        The component of the series that will be modified. Valid options are 'trend' or 'seasonal'.
    fixed_config: dict
        Fixed configuration for the task
    seed: int
        Seed for the random number generator
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following", "reasoning: math"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        target_component_name: str = None,
        fixed_config: dict = None,
        seed: int = None,
    ):
        assert (
            target_component_name is not None
        ), "The modification parameter must be provided. 'trend' or 'seasonal' are valid options."
        self.target_component_name = target_component_name
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        pass

    def apply_modification(self):
        pass

    def recompose_series(self, modified_component, prediction_length):
        if self.target_component_name == "trend":
            return (
                modified_component
                + self.stl.fit().seasonal[-prediction_length:]
                + self.stl.fit().resid[-prediction_length:]
            )
        elif self.target_component_name == "seasonal":
            return (
                self.stl.fit().trend[-prediction_length:]
                + modified_component
                + self.stl.fit().resid[-prediction_length:]
            )
        elif self.target_component_name == "residual":
            return (
                self.stl.fit().trend[-prediction_length:]
                + self.stl.fit().seasonal[-prediction_length:]
                + modified_component
            )
        else:
            raise ValueError(
                "The modification parameter must be provided. 'trend' or 'seasonal' are valid options."
            )


class STLPredMultiplierTask(STLModifierTask):
    """
    A task where the series is first decomposed into trend, seasonality, and residuals
    using STL decomposition. One component of the series is then multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    Parameters:
    ----------
    target_component_name: str
        The component of the series that will be modified. Valid options are 'trend' or 'seasonal'.
    fixed_config: dict
        Fixed configuration for the task
    seed: int
        Seed for the random number generator
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        target_component_name: str = None,
        fixed_config: dict = None,
        seed: int = None,
    ):
        super().__init__(
            target_component_name=target_component_name,
            fixed_config=fixed_config,
            seed=seed,
        )

    @abstractmethod
    def get_background_context(self):
        pass

    def random_instance(self):
        # load dataset
        datasets = ["electricity_hourly"]
        dataset_name = self.random.choice(datasets)
        dataset = get_dataset(dataset_name, regenerate=False, path=DATA_STORAGE_PATH)

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

        start_idx = self.random.randint(0, metadata.prediction_length - 1)
        duration = self.random.randint(1, metadata.prediction_length - start_idx)
        start_datetime = future_series.index[start_idx]
        end_datetime = future_series.index[start_idx + duration]

        history_series.index = history_series.index.to_timestamp()
        future_series.index = future_series.index.to_timestamp()
        window.index = window.index.to_timestamp()
        ground_truth = future_series.copy()

        # {'B': 5, 'D': 1, 'H': 24, 'M': 12, 'ME': 12, 'Q': 4, 'QE': 4, 'S': 3600, 'T': 1440, 'W': 1, 'h': 24, 'min': 1440, 's': 3600}
        seasonality = get_seasonality(metadata.freq)
        self.stl = STL(window, period=seasonality)

        stl_component = self.get_stl_component(self.target_component_name)

        future_series_component = stl_component[-metadata.prediction_length :]

        modified_component = self.apply_modification(
            start_idx, duration, future_series_component
        )

        future_series = self.recompose_series(
            modified_component, metadata.prediction_length
        )

        scenario = self.get_scenario_context(start_datetime, end_datetime)

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = self.get_constraints_context()
        self.background = self.get_background_context()
        self.scenario = scenario
        self.ground_truth = ground_truth
        self.trend = self.stl.fit().trend
        self.seasonal = self.stl.fit().seasonal
        self.residual = self.stl.fit().resid

    def get_constraints_context(self):
        return None

    def get_scenario_context(self, start_datetime, end_datetime):
        return f"The {self.target_component_name} component of the series will be multiplied by {self.multiplier} between {start_datetime} and {end_datetime}."

    def apply_modification(self, start_idx, duration, component_to_modify):
        """
        Applies the modification to the STL component.
        For now, it's a simple multiplication of the component by a random factor.
        It could be expanded to include other modifications, such as:
        - adding a constant value
        - modifying the slope of the trend
        - messing with the frequency of the seasonal component
        """
        modified_component = component_to_modify.copy()
        self.multiplier = self.sample_multiplier()
        modified_component.iloc[start_idx : start_idx + duration] *= self.multiplier

        return modified_component

    def sample_multiplier(self, multiplier_min=-1, multiplier_max=1):
        # pick trend modification from uniform distribution between -1 and 1
        multiplier = self.random.uniform(multiplier_min, multiplier_max)

        return multiplier

    def get_stl_component(self, component):
        if component == "trend":
            return self.stl.fit().trend
        elif component == "seasonal":
            return self.stl.fit().seasonal
        elif component == "residual":
            return self.stl.fit().resid
        else:
            raise ValueError(
                "The modification parameter must be provided. 'trend' or 'seasonal' are valid options."
            )


class STLPredTrendMultiplierTask(STLPredMultiplierTask):
    """
    A task where the trend component of the series is multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(
            target_component_name="trend", fixed_config=fixed_config, seed=seed
        )


class STLPredSeasonalMultiplierTask(STLPredMultiplierTask):
    """
    A task where the seasonal component of the series is multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(
            target_component_name="seasonal", fixed_config=fixed_config, seed=seed
        )


class STLPredResidualMultiplierTask(STLPredMultiplierTask):
    """
    A task where the residual component of the series is multiplied by a random factor.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(
            target_component_name="residual", fixed_config=fixed_config, seed=seed
        )


class STLPredTrendMultiplierWithNoDescriptionTask(STLPredTrendMultiplierTask):
    """
    A task where the trend component of the series is multiplied by a random factor.
    No description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLNoDescriptionContext.get_background_context()


class STLPredTrendMultiplierWithShortDescriptionTask(STLPredTrendMultiplierTask):
    """
    A task where the trend component of the series is multiplied by a random factor.
    A short description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLShortDescriptionContext.get_background_context()


class STLPredTrendMultiplierWithMediumDescriptionTask(STLPredTrendMultiplierTask):
    """
    A task where the trend component of the series is multiplied by a random factor.
    A medium description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLMediumDescriptionContext.get_background_context()


class STLPredTrendMultiplierWithLongDescriptionTask(STLPredTrendMultiplierTask):
    """
    A task where the trend component of the series is multiplied by a random factor.
    A long description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLLongDescriptionContext.get_background_context()


class STLPredSeasonalMultiplierWithNoDescriptionTask(STLPredSeasonalMultiplierTask):
    """
    A task where the seasonal component of the series is multiplied by a random factor.
    No description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLNoDescriptionContext.get_background_context()


class STLPredSeasonalMultiplierWithShortDescriptionTask(STLPredSeasonalMultiplierTask):
    """
    A task where the seasonal component of the series is multiplied by a random factor.
    A short description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLShortDescriptionContext.get_background_context()


class STLPredSeasonalMultiplierWithMediumDescriptionTask(STLPredSeasonalMultiplierTask):
    """
    A task where the seasonal component of the series is multiplied by a random factor.
    A medium description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLMediumDescriptionContext.get_background_context()


class STLPredSeasonalMultiplierWithLongDescriptionTask(STLPredSeasonalMultiplierTask):
    """
    A task where the seasonal component of the series is multiplied by a random factor.
    A long description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLLongDescriptionContext.get_background_context()


class STLPredResidualMultiplierWithNoDescriptionTask(STLPredResidualMultiplierTask):
    """
    A task where the residual component of the series is multiplied by a random factor.
    No description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLNoDescriptionContext.get_background_context()


class STLPredResidualMultiplierWithShortDescriptionTask(STLPredResidualMultiplierTask):
    """
    A task where the residual component of the series is multiplied by a random factor.
    A short description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLShortDescriptionContext.get_background_context()


class STLPredResidualMultiplierWithMediumDescriptionTask(STLPredResidualMultiplierTask):
    """
    A task where the residual component of the series is multiplied by a random factor.
    A medium description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLMediumDescriptionContext.get_background_context()


class STLPredResidualMultiplierWithLongDescriptionTask(STLPredResidualMultiplierTask):
    """
    A task where the residual component of the series is multiplied by a random factor.
    A long description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLLongDescriptionContext.get_background_context()


class STLPredTrendRemovedTask(STLPredTrendMultiplierTask):
    """
    A task where the trend component of the series is removed.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def sample_multiplier(self, multiplier_min=-1, multiplier_max=1):
        return 0

    def get_scenario_context(self, start_datetime, end_datetime):
        return (
            super()
            .get_scenario_context(start_datetime, end_datetime)
            .replace("multiplied by 0", "removed")
        )


class STLPredSeasonalRemovedTask(STLPredSeasonalMultiplierTask):
    """
    A task where the seasonal component of the series is removed.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def sample_multiplier(self, multiplier_min=-1, multiplier_max=1):
        return 0

    def get_scenario_context(self, start_datetime, end_datetime):
        return (
            super()
            .get_scenario_context(start_datetime, end_datetime)
            .replace("multiplied by 0", "removed")
        )


class STLPredResidualRemovedTask(STLPredResidualMultiplierTask):
    """
    A task where the residual component of the series is removed.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def sample_multiplier(self, multiplier_min=-1, multiplier_max=1):
        return 0

    def get_scenario_context(self, start_datetime, end_datetime):
        return (
            super()
            .get_scenario_context(start_datetime, end_datetime)
            .replace("multiplied by 0", "removed")
        )


class STLPredTrendRemovedWithNoDescriptionTask(STLPredTrendRemovedTask):
    """
    A task where the trend component of the series is removed.
    No description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLNoDescriptionContext.get_background_context()


class STLPredTrendRemovedWithShortDescriptionTask(STLPredTrendRemovedTask):
    """
    A task where the trend component of the series is removed.
    A short description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLShortDescriptionContext.get_background_context()


class STLPredTrendRemovedWithMediumDescriptionTask(STLPredTrendRemovedTask):
    """
    A task where the trend component of the series is removed.
    A medium description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLMediumDescriptionContext.get_background_context()


class STLPredTrendRemovedWithLongDescriptionTask(STLPredTrendRemovedTask):
    """
    A task where the trend component of the series is removed.
    A long description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLLongDescriptionContext.get_background_context()


class STLPredSeasonalRemovedWithNoDescriptionTask(STLPredSeasonalRemovedTask):
    """
    A task where the seasonal component of the series is removed.
    No description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLNoDescriptionContext.get_background_context()


class STLPredSeasonalRemovedWithShortDescriptionTask(STLPredSeasonalRemovedTask):
    """
    A task where the seasonal component of the series is removed.
    A short description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLShortDescriptionContext.get_background_context()


class STLPredSeasonalRemovedWithMediumDescriptionTask(STLPredSeasonalRemovedTask):
    """
    A task where the seasonal component of the series is removed.
    A medium description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLMediumDescriptionContext.get_background_context()


class STLPredSeasonalRemovedWithLongDescriptionTask(STLPredSeasonalRemovedTask):
    """
    A task where the seasonal component of the series is removed.
    A long description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLLongDescriptionContext.get_background_context()


class STLPredResidualRemovedWithNoDescriptionTask(STLPredResidualRemovedTask):
    """
    A task where the residual component of the series is removed.
    No description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLNoDescriptionContext.get_background_context()


class STLPredResidualRemovedWithShortDescriptionTask(STLPredResidualRemovedTask):
    """
    A task where the residual component of the series is removed.
    A short description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLShortDescriptionContext.get_background_context()


class STLPredResidualRemovedWithMediumDescriptionTask(STLPredResidualRemovedTask):
    """
    A task where the residual component of the series is removed.
    A medium description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLMediumDescriptionContext.get_background_context()


class STLPredResidualRemovedWithLongDescriptionTask(STLPredResidualRemovedTask):
    """
    A task where the residual component of the series is removed.
    A long description of the STL decomposition is provided.
    Time series: agnostic
    Context: synthetic
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def get_background_context(self):
        return STLLongDescriptionContext.get_background_context()


__TASKS__ = [
    STLPredTrendMultiplierWithNoDescriptionTask,
    STLPredTrendMultiplierWithShortDescriptionTask,
    STLPredTrendMultiplierWithMediumDescriptionTask,
    STLPredTrendMultiplierWithLongDescriptionTask,
    STLPredSeasonalMultiplierWithNoDescriptionTask,
    STLPredSeasonalMultiplierWithShortDescriptionTask,
    STLPredSeasonalMultiplierWithMediumDescriptionTask,
    STLPredSeasonalMultiplierWithLongDescriptionTask,
    # STLPredResidualMultiplierWithNoDescriptionTask,
    # STLPredResidualMultiplierWithShortDescriptionTask,
    # STLPredResidualMultiplierWithMediumDescriptionTask,
    # STLPredResidualMultiplierWithLongDescriptionTask,
    STLPredTrendRemovedWithNoDescriptionTask,
    STLPredTrendRemovedWithShortDescriptionTask,
    STLPredTrendRemovedWithMediumDescriptionTask,
    STLPredTrendRemovedWithLongDescriptionTask,
    STLPredSeasonalRemovedWithNoDescriptionTask,
    STLPredSeasonalRemovedWithShortDescriptionTask,
    STLPredSeasonalRemovedWithMediumDescriptionTask,
    STLPredSeasonalRemovedWithLongDescriptionTask,
    # STLPredResidualRemovedWithNoDescriptionTask,
    # STLPredResidualRemovedWithShortDescriptionTask,
    # STLPredResidualRemovedWithMediumDescriptionTask,
    # STLPredResidualRemovedWithLongDescriptionTask,
]
