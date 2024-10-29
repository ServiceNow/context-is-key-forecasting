"""
Base classes for the benchmark

"""

import numpy as np
import pandas as pd
from typing import Optional
import statsmodels.tsa.tsatools

from abc import ABC, abstractmethod

from .metrics.roi_metric import threshold_weighted_crps
from .metrics.scaling_cache import DefaultScalingCache
from .utils.plot import plot_task
from .config import COMPUTE_METRIC_VARIANCE


ALLOWED_CONTEXT_SOURCES = ["c_h", "c_i", "c_f", "c_cov", "c_causal"]
ALLOWED_SKILLS = [
    "forecasting",
    "natural language processing",
    "instruction following",
    "retrieval: context",
    "retrieval: memory",
    "reasoning: analogy",
    "reasoning: deduction",
    "reasoning: math",
    "reasoning: causal",
]


class BaseTask(ABC):
    """
    Base class for a task

    Parameters:
    -----------
    seed: int
        Seed for the random number generator
    fixed_config: dict
        Fixed configuration for the task

    """

    _context_sources = []
    _skills = ["forecasting", "natural language processing"]
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None):
        self.random = np.random.RandomState(seed)

        # Instantiate task parameters
        if fixed_config is not None:
            self.past_time = fixed_config["past_time"]
            self.future_time = fixed_config["future_time"]
            self.constraints = fixed_config["constraints"]
            self.background = fixed_config["background"]
            self.scenario = fixed_config["scenario"]
        else:
            self.constraints = None
            self.background = None
            self.scenario = None
            self.random_instance()

        config_errors = self.verify_config()
        if config_errors:
            raise RuntimeError(
                f"Incorrect config for {self.__class__.__name__}: {config_errors}"
            )

    @property
    def name(self) -> str:
        """
        Give the name of the task, for reporting purpose

        Returns:
        --------
        name: str
            The name of the task
        """
        return self.__class__.__name__

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # By default, uses the frequency of the data to guess the period.
        # This should be overriden for tasks for which this guess fails.
        freq = self.past_time.index.freq
        if not freq:
            freq = pd.infer_freq(self.past_time.index)
        period = statsmodels.tsa.tsatools.freq_to_period(freq)
        return period

    def verify_config(self) -> list[str]:
        """
        Check whether the task satisfy the correct format for its parameters.

        Returns:
        --------
        errors: list[str]
            A list of textual descriptions of all errors in the format
        """
        errors = []
        # A few tests to make sure that all tasks use a compatible format for the parameters
        # Note: Only the parameters which are used elsewhere are current tested
        if not isinstance(self.past_time, pd.DataFrame):
            errors.append(
                f"past_time is not a pd.DataFrame, but a {self.past_time.__class__.__name__}"
            )
        if not isinstance(self.future_time, pd.DataFrame):
            errors.append(
                f"future_time is not a pd.DataFrame, but a {self.future_time.__class__.__name__}"
            )
        # ... check that the context sources are valid
        for source in self._context_sources:
            if source not in ALLOWED_CONTEXT_SOURCES:
                errors.append(f"Invalid task context source: {source}")
        # ... check that the skills are valid
        for skill in self._skills:
            if skill not in ALLOWED_SKILLS:
                errors.append(f"Invalid task skill: {skill}")
        return errors

    @abstractmethod
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
        pass

    @abstractmethod
    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data

        """
        pass

    def plot(self):
        """
        Plot the task

        Returns:
        --------
        fig: matplotlib.figure.Figure
            The figure containing the plot

        """
        return plot_task(self)

    @property
    def max_directprompt_batch_size(self) -> Optional[int]:
        """
        If set, only request that many samples at once when using a method using Direct Prompting.
        Mainly used to avoid crashing the Llama3-405b server.
        """
        return None


class UnivariateCRPSTask(BaseTask):
    """
    A base class for tasks that require forecasting a single series and that use CRPS for evaluation
    We use the last column of `future_time` as the ground truth for evaluation
    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, seed: int = None, fixed_config: Optional[dict] = None):
        # Instantiate task parameters
        if fixed_config is not None:
            self.region_of_interest = fixed_config["region_of_interest"]
            self.roi_weight = fixed_config["roi_weight"]
            self.metric_constraint = fixed_config["metric_constraint"]
        else:
            # These will be filled during by random_instance(), called in BaseTask.__init__
            self.region_of_interest = None
            self.roi_weight = 0.5
            self.metric_constraint = None

        super().__init__(seed=seed, fixed_config=fixed_config)

    def verify_config(self) -> list[str]:
        """
        Check whether the task satisfy the correct format for its parameters.

        Returns:
        --------
        errors: list[str]
            A list of textual descriptions of all errors in the format
        """
        errors = super().verify_config()
        if self.roi_weight < 0 or self.roi_weight > 1:
            errors.append(f"roi weight ({self.roi_weight}) is not between 0 and 1")
        if self.metric_constraint:
            only_column = self.future_time.columns[-1]
            target = self.future_time[only_column]
            violation = self.metric_constraint.violation(
                samples=target.values[None, :], scaling=1.0
            )[0]
            if violation > 0.0:
                errors.append(
                    f"Constraint {self.metric_constraint} is violated by the ground truth"
                )
        return errors

    def evaluate(self, samples):
        task_scaling = DefaultScalingCache(self.__class__)
        if task_scaling is None:
            return float("nan")

        if len(samples.shape) == 3:
            samples = samples[:, :, 0]

        # This is the dual of pd.Series.to_frame(), compatible with any series name
        only_column = self.future_time.columns[-1]
        target = self.future_time[only_column]
        return threshold_weighted_crps(
            target=target,
            forecast=samples,
            scaling=task_scaling,
            region_of_interest=self.region_of_interest,
            roi_weight=self.roi_weight,
            constraint=self.metric_constraint,
            compute_variance=COMPUTE_METRIC_VARIANCE,
        )
