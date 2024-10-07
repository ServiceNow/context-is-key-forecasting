import scipy
import numpy as np

from .baselines.lag_llama import (
    get_lag_llama_predictions,
    prepare_dataset,
    format_llama_predictions,
)
from .baselines.utils import torch_default_device

from .baselines.statsmodels import (
    ExponentialSmoothingForecaster,
    ETSModelForecaster,
)

SEED = 42


def intersection_over_union_is_low(history_series, future_series, threshold=0.33):
    """
    Check if the intersection over union is low between the range of the history and the range of the future.
    This only works if the series have no trend.
    https://link.springer.com/article/10.1007/s11018-023-02180-2
    Parameters:
    -----------
    history_series: pd.Series
        Historical data
    future_series: pd.Series
        Future data
    Returns:
    --------
    is_low: bool
        True if the overlap is low, False otherwise
    """
    min_upper_bound = min(history_series.max(), future_series.max())
    max_lower_bound = max(history_series.min(), future_series.min())

    max_upper_bound = max(history_series.max(), future_series.max())
    min_lower_bound = min(history_series.min(), future_series.min())

    intersection = min_upper_bound - max_lower_bound
    union = max_upper_bound - min_lower_bound

    intersection_over_union = intersection / union if union != 0 else 0

    return intersection_over_union < threshold


def quartile_intersection_over_union_is_low(
    history_series, future_series, threshold=0.33
):
    """
    Check if the intersection over union is low between the interquartile range of the history and future series.
    This avoids outliers by considering the 25th and 75th quartiles.

    Parameters:
    -----------
    history_series: pd.Series
        Historical data
    future_series: pd.Series
        Future data
    Returns:
    --------
    is_low: bool
        True if the overlap is low, False otherwise
    """
    history_q25, history_q75 = history_series.quantile(0.25), history_series.quantile(
        0.75
    )
    future_q25, future_q75 = future_series.quantile(0.25), future_series.quantile(0.75)

    min_upper_bound = min(history_q75, future_q75)
    max_lower_bound = max(history_q25, future_q25)

    max_upper_bound = max(history_q75, future_q75)
    min_lower_bound = min(history_q25, future_q25)

    intersection = min_upper_bound - max_lower_bound
    union = max_upper_bound - min_lower_bound

    intersection_over_union = intersection / union if union != 0 else 0

    return intersection_over_union < threshold


def median_absolute_deviation_intersection_is_low(
    history_series, future_series, threshold=0.33
):
    """
    Check if the intersection over union is low between the median and the range based on the median absolute deviation of the history and future series.
    This method is robust to outliers.

    Parameters:
    -----------
    history_series: pd.Series
        Historical data
    future_series: pd.Series
        Future data
    Returns:
    --------
    is_low: bool
        True if the overlap is low, False otherwise
    """

    def median_absolute_deviation_interval(series):
        """
        Calculate the interval around the median using the median absolute deviation (MAD).

        Parameters:
        -----------
        series: pd.Series
            Input data series.

        Returns:
        --------
        lower_bound: float
            Lower bound of the interval.
        upper_bound: float
            Upper bound of the interval.

        Note:
        -----
        The constant 1.4826 scales MAD to match the standard deviation for a normal distribution.
        """
        median = series.median()
        mad = (series - median).abs().median()
        lower_bound = median - 1.4826 * mad
        upper_bound = median + 1.4826 * mad
        return lower_bound, upper_bound

    history_lower, history_upper = median_absolute_deviation_interval(history_series)
    future_lower, future_upper = median_absolute_deviation_interval(future_series)

    min_upper_bound = min(history_upper, future_upper)
    max_lower_bound = max(history_lower, future_lower)

    max_upper_bound = max(history_upper, future_upper)
    min_lower_bound = min(history_lower, future_lower)

    intersection = min_upper_bound - max_lower_bound
    union = max_upper_bound - min_lower_bound

    intersection_over_union = intersection / union if union != 0 else 0

    return intersection_over_union < threshold


def is_baseline_prediction_poor(
    history_series,
    future_series,
    constraints,
    period=None,
    n_samples=50,
    constraint_satisfaction_threshold=0.2,
    baselines=None,
    baseline_evaluation_criteria=None,
):
    """
    Check if the baseline prediction is bad.

    Parameters:
    - history_series: Series of historical data.
    - future_series: Series of future data.
    - constraints: Constraints to evaluate predictions against.
    - n_samples: Number of samples for forecasting.
    - baselines: List of baseline models to evaluate. Default is ["exponential_smoothing"].
        Current baselines are:
        - "lag-llama": Lag-Llama model
        - "exponential_smoothing": Exponential Smoothing model
        - "ets": ETS model
    - baseline_evaluation_criteria: "all" to check if all baselines satisfy the condition, "any" to check if any baseline satisfies the condition.
    """
    # if baselines is a single element, put it in a list
    baselines = baselines
    if isinstance(baselines, str):
        baselines = [baselines]
    if baselines is None:
        baselines = ["exponential_smoothing"]

    if baseline_evaluation_criteria is None:
        baseline_evaluation_criteria = "all"

    baseline_forecasts = []

    all_satisfy = True
    any_satisfies = False

    for baseline in baselines:
        forecasts = get_baseline_forecasts(
            history_series, future_series, n_samples, baseline, period
        )
        baseline_forecasts.append(forecasts)
        constraint_satisfaction_rate = get_constraint_satisfaction_rate(
            forecasts, constraints
        )

        if baseline_evaluation_criteria == "all":
            if constraint_satisfaction_rate >= constraint_satisfaction_threshold:
                all_satisfy = False
                break  # not all baselines satisfy the condition
        elif baseline_evaluation_criteria == "any":
            if constraint_satisfaction_rate < constraint_satisfaction_threshold:
                any_satisfies = True
                break  # at least one baseline satisfies the condition

    if baseline_evaluation_criteria == "all":
        return all_satisfy
    elif baseline_evaluation_criteria == "any":
        return any_satisfies
    else:
        raise ValueError(f"Unknown mode: {baseline_evaluation_criteria}")


def get_baseline_forecasts(history_series, future_series, n_samples, baseline, period):
    """
    Get the forecasts from a baseline model.
    Current baselines are:
    - "lag-llama": Lag-Llama model
    - "exponential_smoothing": Exponential Smoothing model
    - "ets": ETS model
    Parameters:
    -----------
    history_series: pd.Series
        Historical data
    future_series: pd.Series
        Future data
    n_samples: int
        Number of samples for forecasting
    baseline: str
        Baseline model to use
    """
    if baseline == "lag-llama":
        dataset = prepare_dataset(history_series.to_frame(), future_series.to_frame())
        # Get the forecast from the Lag-Llama model
        forecasts, _ = get_lag_llama_predictions(
            dataset=dataset,
            prediction_length=len(future_series),
            device=torch_default_device(),
            num_samples=n_samples,
            seed=SEED,
        )
        forecasts = format_llama_predictions(forecasts, dtype=history_series.dtype)
    elif baseline == "exponential_smoothing":
        if period is None:
            raise ValueError("Period must be provided for Exponential Smoothing")
        forecaster = ExponentialSmoothingForecaster()
        forecasts = forecaster.forecast(
            history_series.to_frame(),
            future_series.to_frame(),
            seasonal_periods=period,
            n_samples=n_samples,
        )
    elif baseline == "ets":
        if period is None:
            raise ValueError("Period must be provided for ETS")
        forecaster = ETSModelForecaster()
        forecasts = forecaster.forecast(
            history_series.to_frame(),
            future_series.to_frame(),
            seasonal_periods=period,
            n_samples=n_samples,
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return forecasts


def get_constraint_satisfaction_rate(samples, constraints=None):
    """
    Since we don't know the ground truth distribution, the evaluation should
    be done by comparing the forecast with the constraints
    The score is the proportion of samples that satisfy each constraint,
    averaged over all constraints.
    As a side-effect, sets the attribute prop_satisfied_constraints to the
    proportion of samples that satisfy each constraint.
    Parameters:
    -----------
    samples: np.ndarray (num_samples, num_timesteps) or (num_samples, num_timesteps, 1)
        Samples from the inferred distribution
    Returns:
    --------
    prop_satisfied_constraint: float
        Proportion of samples that satisfy the constraints
    """
    if constraints is None:
        raise ValueError("No constraints provided")

    if len(samples.shape) == 3:
        samples = samples[:, :, 0]  # (n_samples, n_time)

    prop_satisfied_constraints = {}
    for constraint, value in constraints.items():
        if constraint == "min":
            good_samples = np.all(samples >= value, axis=1)
        elif constraint == "max":
            good_samples = np.all(samples <= value, axis=1)

        prop_satisfied_constraint = np.mean(good_samples)
        prop_satisfied_constraints[constraint] = prop_satisfied_constraint

    constraint_satisfaction_rate = np.mean(
        np.array(list(prop_satisfied_constraints.values()))
    )
    # prop_satisfied_constraints["satisfaction_rate"] = constraint_satisfaction_rate
    # prop_satisfied_constraints = prop_satisfied_constraints

    return constraint_satisfaction_rate
