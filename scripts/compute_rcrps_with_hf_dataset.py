"""
Copyright 2025 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# This code is an adaptation of
# https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/metrics/roi_metric.py
# to make it convenient to use with the Hugging Face version of the Context-is-Key benchmark.
# Please see the __main__ section for an example of how to use it.

import numpy as np
import pandas as pd
from io import StringIO
from datasets import Dataset
from fractions import Fraction


def crps(
    target: np.array,
    samples: np.array,
) -> np.array:
    """
    Compute the CRPS using the probability weighted moment form.
    See Eq ePWM from "Estimation of the Continuous Ranked Probability Score with
    Limited Information and Applications to Ensemble Weather Forecasts"
    https://link.springer.com/article/10.1007/s11004-017-9709-7

    This is a O(n log n) per variable exact implementation, without estimation bias.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (variable dimensions)
    samples: np.ndarray
        The forecast values. (n_samples, variable dimensions)

    Returns:
    --------
    crps: np.ndarray
        The CRPS for each of the (variable dimensions)
    """
    assert (
        target.shape == samples.shape[1:]
    ), f"shapes mismatch between: {target.shape} and {samples.shape}"

    num_samples = samples.shape[0]
    num_dims = samples.ndim
    sorted_samples = np.sort(samples, axis=0)

    abs_diff = (
        np.abs(np.expand_dims(target, axis=0) - sorted_samples).sum(axis=0)
        / num_samples
    )

    beta0 = sorted_samples.sum(axis=0) / num_samples

    # An array from 0 to num_samples - 1, but expanded to allow broadcasting over the variable dimensions
    i_array = np.expand_dims(np.arange(num_samples), axis=tuple(range(1, num_dims)))
    beta1 = (i_array * sorted_samples).sum(axis=0) / (num_samples * (num_samples - 1))

    return abs_diff + beta0 - 2 * beta1


def _crps_ea_Xy_eb_Xy(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E|Xa - ya| * E|Xb' - yb|
    """
    N = len(Xa)
    result = 0.0
    product = np.abs(Xa[:, None] - ya) * np.abs(Xb[None, :] - yb)  # i, j
    i, j = np.diag_indices(N)
    product[i, j] = 0
    result = product.sum()
    return result / (N * (N - 1))


def _crps_ea_XX_eb_XX(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E|Xa - Xa'| * E|Xb'' - Xb'''|
    """
    N = len(Xa)

    # We want to compute:
    # sum_i≠j≠k≠l |Xa_i - Xa_j| |Xb_k - Xb_l|
    # Instead of doing a sum over i, j, k, l all differents,
    # we take the sum over all i, j, k, l (which is the product between a sum over i, j and a sum over k, l),
    # then substract the collisions, ignoring those between i and j and those between k and l, since those
    # automatically gives zero.

    sum_ea_XX = np.abs(Xa[:, None] - Xa[None, :]).sum()
    sum_eb_XX = np.abs(Xb[:, None] - Xb[None, :]).sum()

    # Single conflicts: either i=k, i=l, j=k, or j=l
    # By symmetry, we are left with: 4 sum_i≠j≠k |Xa_i - Xa_j| |Xb_i - Xb_k|
    left = np.abs(Xa[:, None, None] - Xa[None, :, None])  # i, j, k
    right = np.abs(Xb[:, None, None] - Xb[None, None, :])  # i, j, k
    product = left * right
    j, k = np.diag_indices(N)
    product[:, j, k] = 0
    sum_single_conflict = product.sum()

    # Double conflicts: either i=k and j=l, or i=l and j=k
    # By symmetry, we are left with: 2 sum_i≠j |Xa_i - Xa_j| |Xb_i - Xb_j|
    left = np.abs(Xa[:, None] - Xa[None, :])  # i, j
    right = np.abs(Xb[:, None] - Xb[None, :])  # i, j
    product = left * right
    sum_double_conflict = product.sum()

    result = sum_ea_XX * sum_eb_XX - 4 * sum_single_conflict - 2 * sum_double_conflict
    return result / (N * (N - 1) * (N - 2) * (N - 3))


def _crps_ea_Xy_eb_XX(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E|Xa - ya| * E|Xb' - Xb''|
    """
    N = len(Xa)

    left = np.abs(Xa[:, None, None] - ya)  # i, j, k
    right = np.abs(Xb[None, :, None] - Xb[None, None, :])  # i, j, k
    product = left * right
    i, j = np.diag_indices(N)
    product[i, j, :] = 0
    i, k = np.diag_indices(N)
    product[i, :, k] = 0
    result = product.sum()
    return result / (N * (N - 1) * (N - 2))


def _crps_f_Xy(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E(|Xa - ya| * |Xb - yb|)
    """
    N = len(Xa)
    product = np.abs(Xa - ya) * np.abs(Xb - yb)  # i
    result = product.sum()
    return result / N


def _crps_f_XXXy(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E(|Xa - Xa'| * |Xb - yb|)
    """
    N = len(Xa)
    left = np.abs(Xa[:, None] - Xa[None, :])  # i, j
    right = np.abs(Xb[:, None] - yb)  # i, j
    product = left * right
    result = product.sum()
    return result / (N * (N - 1))


def _crps_f_XX(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E(|Xa - Xa'| * |Xb - Xb'|)
    """
    N = len(Xa)
    left = np.abs(Xa[:, None] - Xa[None, :])  # i, j
    right = np.abs(Xb[:, None] - Xb[None, :])  # i, j
    product = left * right
    result = product.sum()
    return result / (N * (N - 1))


def _crps_f_XXXX(Xa, ya, Xb, yb):
    """
    Unbiased estimate of:
    E(|Xa - Xa'| * |Xb - Xb''|)
    """
    N = len(Xa)
    left = np.abs(Xa[:, None, None] - Xa[None, :, None])  # i, j, k
    right = np.abs(Xb[:, None, None] - Xb[None, None, :])  # i, j, k
    product = left * right
    j, k = np.diag_indices(N)
    product[:, j, k] = 0
    result = product.sum()
    return result / (N * (N - 1) * (N - 2))


def crps_covariance(
    Xa: np.array,
    ya: float,
    Xb: np.array,
    yb: float,
) -> float:
    """
    Unbiased estimate of the covariance between the CRPS of two correlated random variables.
    If Xa == Xb and ya == yb, returns the variance of the CRPS instead.

    Parameters:
    -----------
    Xa: np.ndarray
        Samples from a forecast for the first variable. (n_samples)
    ya: float
        The ground-truth value for the first variable.
    Xb: np.ndarray
        Samples from a forecast for the second variable. (n_samples)
    yb: float
        The ground-truth value for the second variable.

    Returns:
    --------
    covariance: float
        The covariance between the CRPS estimators.
    """
    N = len(Xa)

    ea_Xy_eb_Xy = _crps_ea_Xy_eb_Xy(Xa, ya, Xb, yb)
    ea_Xy_eb_XX = _crps_ea_Xy_eb_XX(Xa, ya, Xb, yb)
    ea_XX_eb_Xy = _crps_ea_Xy_eb_XX(Xb, yb, Xa, ya)
    ea_XX_eb_XX = _crps_ea_XX_eb_XX(Xa, ya, Xb, yb)

    f_Xy = _crps_f_Xy(Xa, ya, Xb, yb)
    f_XXXy = _crps_f_XXXy(Xa, ya, Xb, yb)
    f_XyXX = _crps_f_XXXy(Xb, yb, Xa, ya)
    f_XX = _crps_f_XX(Xa, ya, Xb, yb)
    f_XXXX = _crps_f_XXXX(Xa, ya, Xb, yb)

    return (
        -(1 / N) * ea_Xy_eb_Xy
        + (1 / N) * ea_Xy_eb_XX
        + (1 / N) * ea_XX_eb_Xy
        - ((2 * N - 3) / (2 * N * (N - 1))) * ea_XX_eb_XX
        + (1 / N) * f_Xy
        - (1 / N) * f_XXXy
        - (1 / N) * f_XyXX
        + (1 / (2 * N * (N - 1))) * f_XX
        + ((N - 2) / (N * (N - 1))) * f_XXXX
    )


def weighted_sum_crps_variance(
    target: np.array,
    samples: np.array,
    weights: np.array,
) -> float:
    """
    Unbiased estimator of the variance of the numerical estimate of the
    given weighted sum of CRPS values.

    This implementation assumes that the univariate is estimated using:
    CRPS(X, y) ~ (1 / n) * sum_i |x_i - y| - 1 / (2 * n * (n-1)) * sum_i,i' |x_i - x_i'|.
    This formula gives the same result as the one used in the crps() implementation above.

    Note that this is a heavy computation, being O(k^2 n^3) with k variables and n samples.
    Also, while it is unbiased, it is not guaranteed to be >= 0.

    Parameters:
    -----------
    target: np.ndarray
        The target values: y in the above formula. (k variables)
    samples: np.ndarray
        The forecast values: X in the above formula. (n samples, k variables)
    weights: np.array
        The weight given to the CRPS of each variable. (k variables)

    Returns:
    --------
    variance: float
        The variance of the weighted sum of the CRPS estimators.
    """
    assert len(target.shape) == 1
    assert len(samples.shape) == 2
    assert len(weights.shape) == 1
    assert target.shape[0] == samples.shape[1] == weights.shape[0]

    s = 0.0

    for i in range(target.shape[0]):
        for j in range(i, target.shape[0]):
            Xa = samples[:, i]
            Xb = samples[:, j]
            ya = target[i]
            yb = target[j]

            if i == j:
                s += weights[i] * weights[j] * crps_covariance(Xa, ya, Xb, yb)
            else:
                # Multiply by 2 since we would get the same results by switching i and j
                s += 2 * weights[i] * weights[j] * crps_covariance(Xa, ya, Xb, yb)

    return s


def mean_crps(target, samples):
    """
    The mean of the CRPS over all variables
    """
    if target.size > 0:
        return crps(target, samples).mean()
    else:
        raise RuntimeError(
            f"CRPS received an empty target. Shapes = {target.shape} and {samples.shape}"
        )


def compute_constraint_violation(
    entry: dict,
    samples: np.array,
    scaling: float,
) -> float:
    violation = 0.0
    scaled_samples = scaling * samples

    # Min constraint
    scaled_threshold = scaling * entry["constraint_min"]
    violation += (scaled_threshold - scaled_samples).clip(min=0).mean(axis=1)

    # Max constraint
    scaled_threshold = scaling * entry["constraint_max"]
    violation += (scaled_samples - scaled_threshold).clip(min=0).mean(axis=1)

    # Variable max constraint
    if len(entry["constraint_variable_max_index"]) > 0:
        indexed_samples = scaled_samples[:, entry["constraint_variable_max_index"]]
        scaled_thresholds = scaling * np.array(entry["constraint_variable_max_values"])
        violation += (
            (indexed_samples - scaled_thresholds[None, :]).clip(min=0).mean(axis=1)
        )

    return violation


def roi_crps(
    entry: dict,
    forecast: np.array,
) -> dict[str, float]:
    """
    Compute the Region-of-Interest CRPS for a single entry of the context-is-key Hugging Face dataset,
    for the given forecast.

    Parameters:
    ----------
    entry: dict
        A dictionary containing a single entry of the context-is-key Hugging Face dataset.
    forecast: np.array
        The forecast values. (n_samples, n_timesteps)

    Returns:
    --------
    result: dict[str, float]
        A dictionary containing the following entries:
        "metric": the final metric.
        "raw_metric": the metric before the log transformation.
        "scaling": the scaling factor applied to the CRPS and the violations.
        "crps": the weighted CRPS.
        "roi_crps": the CRPS only for the region of interest.
        "non_roi_crps": the CRPS only for the forecast not in the region of interest.
        "violation_mean": the average constraint violation over the samples.
        "violation_crps": the CRPS of the constraint violation.
        "metric_variance": an unbiased estimate of the variance of the metric.
    """
    future_time = pd.read_json(StringIO(entry["future_time"]))
    target = future_time[future_time.columns[-1]].to_numpy()

    assert (
        future_time.shape[0] == forecast.shape[1]
    ), "Incorrect number of timesteps in forecast"

    variance_target = target.to_numpy() if isinstance(target, pd.Series) else target
    variance_forecast = forecast

    if entry["region_of_interest"]:
        roi_mask = np.zeros(forecast.shape[1], dtype=bool)
        for i in entry["region_of_interest"]:
            roi_mask[i] = True

        roi_crps = mean_crps(target=target[roi_mask], samples=forecast[:, roi_mask])
        non_roi_crps = mean_crps(
            target=target[~roi_mask], samples=forecast[:, ~roi_mask]
        )
        crps_value = 0.5 * roi_crps + 0.5 * non_roi_crps
        standard_crps = mean_crps(target=target, samples=forecast)
        num_roi_timesteps = roi_mask.sum()
        num_non_roi_timesteps = (~roi_mask).sum()
        variance_weights = entry["metric_scaling"] * (
            0.5 * roi_mask / num_roi_timesteps
            + (1 - 0.5) * ~roi_mask / num_non_roi_timesteps
        )
    else:
        crps_value = mean_crps(target=target, samples=forecast)
        # Those will only be used in the reporting
        roi_crps = crps_value
        non_roi_crps = crps_value
        standard_crps = crps_value
        num_roi_timesteps = len(target)
        num_non_roi_timesteps = 0
        variance_weights = np.full(
            target.shape, fill_value=entry["metric_scaling"] / len(target)
        )

    violation_amount = compute_constraint_violation(
        entry, samples=forecast, scaling=entry["metric_scaling"]
    )
    violation_func = 10.0 * violation_amount

    # The target is set to zero, since we make sure that the ground truth always satisfy the constraints
    # The crps code assume multivariate input, so add a dummy dimension
    violation_crps = crps(target=np.zeros(1), samples=violation_func[:, None])[0]

    variance_target = np.concatenate((variance_target, np.zeros(1)), axis=0)
    variance_forecast = np.concatenate(
        (variance_forecast, violation_func[:, None]), axis=1
    )
    variance_weights = np.concatenate((variance_weights, 1.0 * np.ones(1)), axis=0)

    raw_metric = entry["metric_scaling"] * crps_value + violation_crps
    metric = raw_metric

    # Computing the variance of the RCPRS is much more expensive,
    # especially when the number of samples is large.
    # So it can be commented out if not desired.
    variance = weighted_sum_crps_variance(
        target=variance_target,
        samples=variance_forecast,
        weights=variance_weights,
    )

    return {
        "metric": metric,
        "raw_metric": raw_metric,
        "scaling": entry["metric_scaling"],
        "crps": entry["metric_scaling"] * crps_value,
        "roi_crps": entry["metric_scaling"] * roi_crps,
        "non_roi_crps": entry["metric_scaling"] * non_roi_crps,
        "standard_crps": entry["metric_scaling"] * standard_crps,
        "num_roi_timesteps": num_roi_timesteps,
        "num_non_roi_timesteps": num_non_roi_timesteps,
        "violation_mean": violation_amount.mean(),
        "violation_crps": violation_crps,
        "variance": variance,
    }


def compute_all_rcprs(
    dataset: Dataset,
    forecasts: list[dict],
) -> tuple[float, float]:
    """
    Compute the Region-of-Interest CRPS for all instances in the Context-is-Key dataset.

    Parameters:
    ----------
    dataset: Dataset
        The Context-is-Key dataset.
    forecasts: list[dict]
        A list of dictionaries, each containing the following keys:
        - "name": the name of the task for which the forecast is made.
        - "seed": the seed of the instance for which the forecast is made.
        - "forecast": the forecast values. (n_samples, n_timesteps)

    Returns:
    --------
    mean_crps: float
        The aggregated RCRPS over all instances.
    std_crps: float
        An estimate of the standard error of the aggregated RCRPS.
    """
    weighted_sum_rcprs = 0.0
    weighted_sum_variance = 0.0
    total_weight = 0.0

    for entry, forecast in zip(dataset, forecasts):
        if entry["name"] != forecast["name"]:
            raise ValueError(
                f"Forecast name {forecast['name']} does not match dataset entry name {entry['name']}"
            )
        if entry["seed"] != forecast["seed"]:
            raise ValueError(
                f"Forecast seed {forecast['seed']} does not match dataset entry seed {entry['seed']}"
            )
        metric_output = roi_crps(
            entry=entry,
            forecast=forecast["forecast"],
        )

        weight = Fraction(entry["weight"])

        # Apply the cap of RCPRS = 5 to the metric
        if metric_output["metric"] >= 5.0:
            metric_output["metric"] = 5.0
            metric_output["variance"] = 0.0

        weighted_sum_rcprs += weight * metric_output["metric"]
        weighted_sum_variance += weight * weight * metric_output["variance"]
        total_weight += weight

    mean_crps = weighted_sum_rcprs / total_weight
    std_crps = np.sqrt(weighted_sum_variance) / total_weight

    return mean_crps, std_crps


if __name__ == "__main__":
    # An example of how to use this function,
    # by using a naive forecaster which use random values from the past as its forecast.

    from datasets import load_dataset

    dataset = load_dataset("ServiceNow/context-is-key", split="test")

    # Create a random forecast for each instance in the dataset
    forecasts = []
    for entry in dataset:
        past_time = pd.read_json(StringIO(entry["past_time"]))
        future_time = pd.read_json(StringIO(entry["future_time"]))
        forecast = {
            "name": entry["name"],
            "seed": entry["seed"],
            "forecast": np.random.choice(
                past_time.to_numpy()[:, -1],
                size=(25, len(future_time)),
                replace=True,
            ),
        }
        forecasts.append(forecast)

    mean_crps, std_crps = compute_all_rcprs(dataset, forecasts)
    print(f"Mean RCRPS: {mean_crps}")
    print(f"Standard error of RCRPS: {std_crps}")
