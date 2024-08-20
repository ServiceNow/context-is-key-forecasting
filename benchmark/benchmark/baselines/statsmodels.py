import pandas as pd
import numpy as np
import statsmodels
import statsmodels.tsa.holtwinters
import statsmodels.tsa.exponential_smoothing.ets
from typing import Literal

from .base import Baseline
from ..base import BaseTask


class ExponentialSmoothingForecaster(Baseline):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        trend: Literal["add", "mul", None] = "add",
        seasonal: Literal["add", "mul", None] = "add",
    ):
        """
        Get predictions from an Exponential Smoothing model.

        Parameters:
        -----------
        trend: ["add", "mul", or None]
            Whether to add a trend component to the forecast.
            If "add", the component is additive, and if "mul", it is multiplicative.
        seasonal: ["add", "mul", or None]
            Whether to add a seasonal component to the forecast.
            If "add", the component is additive, and if "mul", it is multiplicative.

        Notes:
        ------
        This model requires a seasonal periodicity, which it currently get from a
        hard coded association from the data index frequency (hourly -> 24 hours periods).
        """
        super().__init__()

        self.trend = trend
        self.seasonal = seasonal

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        return self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=task_instance.seasonal_period,
            n_samples=n_samples,
        )

    def forecast(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        seasonal_periods: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarly meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.

        Note: If seasonal_periods is <= 0, then the seasonal component is skipped.
        """
        # If there is no period, then disable the seasonal component of the model (seasonal_periods will be ignored)
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=past_time[past_time.columns[-1]],
            trend=self.trend,
            seasonal=self.seasonal if seasonal_periods >= 1 else None,
            seasonal_periods=seasonal_periods,
        )

        result = model.fit()

        simulations = result.simulate(
            nsimulations=future_time.shape[0], repetitions=n_samples
        )

        return simulations.to_numpy().transpose()[..., None]

    @property
    def cache_name(self) -> str:
        args_to_include = ["trend", "seasonal"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )


class ETSModelForecaster(Baseline):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        trend: Literal["add", "mul", None] = "add",
        seasonal: Literal["add", "mul", None] = "add",
        error: Literal["add", "mul"] = "add",
    ):
        """
        Get predictions from an ETS (Error-Trend-Seasonality) model.

        Parameters:
        -----------
        trend: ["add", "mul", or None]
            Whether to add a trend component to the forecast.
            If "add", the component is additive, and if "mul", it is multiplicative.
        seasonal: ["add", "mul", or None]
            Whether to add a seasonal component to the forecast.
            If "add", the component is additive, and if "mul", it is multiplicative.
        error: ["add", "mul"]
            Configuration for the error component to the forecast.
            If "add", the component is additive, and if "mul", it is multiplicative.

        Notes:
        ------
        This model requires a seasonal periodicity, which it currently get from a
        hard coded association from the data index frequency (hourly -> 24 hours periods).
        """
        super().__init__()

        self.trend = trend
        self.seasonal = seasonal
        self.error = error

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        return self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=task_instance.seasonal_period,
            n_samples=n_samples,
        )

    def forecast(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        seasonal_periods: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarly meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.

        Note: If seasonal_periods is <= 0, then the seasonal component is skipped.
        """
        # If there is no period, then disable the seasonal component of the model (seasonal_periods will be ignored)
        model = statsmodels.tsa.exponential_smoothing.ets.ETSModel(
            endog=past_time[past_time.columns[-1]],
            trend=self.trend,
            seasonal=self.seasonal if seasonal_periods >= 1 else None,
            error=self.error,
            seasonal_periods=seasonal_periods,
        )

        # Avoid L-BFGS-B output spam
        result = model.fit(disp=False)

        simulations = result.simulate(
            nsimulations=future_time.shape[0], repetitions=n_samples
        )

        return simulations.to_numpy().transpose()[..., None]

    @property
    def cache_name(self) -> str:
        args_to_include = ["trend", "seasonal", "error"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
