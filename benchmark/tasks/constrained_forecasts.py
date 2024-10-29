"""
Tasks where the output forecast must be censored by some constraint.
Since the exact output distribution can be known, we use the Wasserstein
distance of the forecast at each time point as the metric,
normalized by what you would obtained ignoring the contraint.
"""

import scipy
import numpy as np
import pandas as pd
from abc import abstractmethod

from ..base import BaseTask


class BaseConstrainedTask(BaseTask):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        seed: int = None,
        constraint_less_than: bool = False,
        constraint_value: float = 0.0,
        num_samples: int = 1000,
        num_hist_values: int = 20,
        num_pred_values: int = 10,
    ):
        self.constraint_less_than = constraint_less_than
        self.constraint_value = constraint_value

        self.num_samples = num_samples
        self.num_hist_values = num_hist_values
        self.num_pred_values = num_pred_values

        super().__init__(seed=seed, fixed_config=None)

    @abstractmethod
    def generate_hist(self) -> np.array:
        pass

    @abstractmethod
    def generate_pred(self) -> np.array:
        pass

    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data
        """
        hist_values = self.generate_hist()
        pred_values = self.generate_pred()

        if self.constraint_less_than:
            const_hist_values = hist_values.clip(max=self.constraint_value)
            const_pred_values = pred_values.clip(max=self.constraint_value)
            constraints = f"Value <= {self.constraint_value}"
        else:
            const_hist_values = hist_values.clip(min=self.constraint_value)
            const_pred_values = pred_values.clip(min=self.constraint_value)
            constraints = f"Value >= {self.constraint_value}"

        # Convert the constrained series to Pandas series, for compatibility with what is in misleading_history
        # TODO: Randomly select frequency + starting date
        # inclusive="left" drops the last value from the index
        history_series = pd.Series(
            data=const_hist_values,
            index=pd.date_range(
                end="2010-01-01",
                freq="D",
                periods=self.num_hist_values + 1,
                inclusive="left",
            ),
        ).to_frame()
        future_series = pd.Series(
            data=const_pred_values[0, :],
            index=pd.date_range(
                start="2010-01-01",
                freq="D",
                periods=self.num_pred_values,
                inclusive="both",
            ),
        ).to_frame()

        # Values required for the evaluation,
        # regenerating the non_const forecast to allow measuring the distance between both
        self.perfect_const_forecast = const_pred_values
        self.perfect_non_const_forecast = self.generate_pred()

        # Instantiate the class variables
        self.past_time = history_series
        self.future_time = future_series
        self.constraints = constraints
        self.background = None
        self.scenario = None

    def evaluate(self, samples: np.ndarray):
        """
        Evaluate success based on samples from the inferred distribution

        Parameters:
        -----------
        samples: np.ndarray, shape (n_samples, n_time, n_dim)
            Samples from the inferred distribution

        Returns:
        --------
        metric: float
            Metric value

        """
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]

        cum_metric = 0.0
        cum_naive = 0.0

        for t_idx in range(self.num_pred_values):
            cum_metric += scipy.stats.wasserstein_distance(
                samples[:, t_idx], self.perfect_const_forecast[:, t_idx]
            )
            cum_naive += scipy.stats.wasserstein_distance(
                self.perfect_non_const_forecast[:, t_idx],
                self.perfect_const_forecast[:, t_idx],
            )

        return (cum_metric - cum_naive) / self.num_pred_values


class ConstrainedRandomWalk(BaseConstrainedTask):
    def __init__(
        self,
        seed: int = None,
        variance: float = 1.0,
        trend: float = 0.0,
        start_value: float = 0.0,
        constraint_less_than: bool = False,
        constraint_value: float = 0.0,
        num_hist_values: int = 20,
        num_pred_values: int = 10,
        num_samples: int = 1000,
    ):
        self.variance = variance
        self.trend = trend
        self.start_value = start_value  # Value of the last point in the history
        self.num_hist_values = num_hist_values
        self.num_pred_values = num_pred_values

        super().__init__(
            seed=seed,
            constraint_less_than=constraint_less_than,
            constraint_value=constraint_value,
            num_samples=num_samples,
            num_hist_values=num_hist_values,
            num_pred_values=num_pred_values,
        )

    def generate_hist(self) -> np.array:
        hist_steps = (
            np.sqrt(self.variance) * self.random.randn(self.num_hist_values)
            + self.trend
        )
        hist_values = np.cumsum(hist_steps)
        hist_values = hist_values - hist_values[-1] + self.start_value
        return hist_values

    def generate_pred(self) -> np.array:
        pred_steps = (
            np.sqrt(self.variance)
            * self.random.randn(self.num_samples, self.num_pred_values)
            + self.trend
        )
        pred_values = np.cumsum(pred_steps, axis=1) + self.start_value
        return pred_values

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        return -1


class ConstrainedNoisySine(BaseConstrainedTask):
    def __init__(
        self,
        seed: int = None,
        sine_frequency: float = np.pi / 5,
        sine_phase: float = 0.0,
        sine_amplitude: float = 1.0,
        trend: float = 0.0,
        start_value: float = 0.0,
        noise_amplitude: float = 0.5,
        constraint_less_than: bool = False,
        constraint_value: float = 0.0,
        num_hist_values: int = 20,
        num_pred_values: int = 10,
        num_samples: int = 1000,
    ):
        self.sine_frequency = sine_frequency
        self.sine_phase = sine_phase
        self.sine_amplitude = sine_amplitude
        self.trend = trend
        self.start_value = (
            start_value  # Value of the last point in the history (with no phase)
        )
        self.noise_amplitude = noise_amplitude

        super().__init__(
            seed=seed,
            constraint_less_than=constraint_less_than,
            constraint_value=constraint_value,
            num_samples=num_samples,
            num_hist_values=num_hist_values,
            num_pred_values=num_pred_values,
        )

    def generate_hist(self) -> np.array:
        t = np.arange(-self.num_hist_values + 1, 1)
        base = (
            np.sin(self.sine_frequency * t + self.sine_phase)
            + self.start_value
            + self.trend * t
        )
        hist_values = base + self.noise_amplitude * self.random.randn(
            self.num_hist_values
        )
        return hist_values

    def generate_pred(self) -> np.array:
        t = np.arange(1, self.num_pred_values + 1)
        base = (
            np.sin(self.sine_frequency * t + self.sine_phase)
            + self.start_value
            + self.trend * t
        )
        hist_values = base[None, :] + self.noise_amplitude * self.random.randn(
            self.num_samples, self.num_pred_values
        )
        return hist_values

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # The frequency is not linear
        return -1


__TASKS__ = []  # No tasks currently included in the benchmark
