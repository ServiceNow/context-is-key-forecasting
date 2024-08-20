from typing import Optional
import numpy as np

from .constraints import Constraint
from .crps import crps


def mean_crps(target, samples):
    """
    The mean of the CRPS over all variables
    """
    if target.size > 0:
        return crps(target, samples).mean()
    else:
        return 0.0


def threshold_weighted_crps(
    target: np.array,
    forecast: np.array,
    region_of_interest=None,
    roi_weight: float = 0.5,
    constraint: Optional[Constraint] = None,
    violation_factor: float = 5.0,
    log_barrier: bool = False,
) -> dict[str, float]:
    """
    Compute the scaled twCRPS, which adds a penalty term when constraints are violated.

    Without scaling and region of interest, the twCRPS is defined as:
    twCRPS(X, y) = CRPS(v(X), v(y)),
    where the multivariate CRPS is computed as the sum of the univariate CRPS over all dimensions,
    and v(z) is an arbitrary transform.
    In our case, we select the transform to be:
    v(z) = [z / length(z), exp(violation_factor * constraint violation) - 1].

    For the scaling, we divide the CRPS by the range of values in the target.

    For the region of interest, we compute the mean CRPS individually on both the region of interest
    and the rest of the forecast, which we then combine using the given weight.

    Parameters:
    ----------
    target: np.array
        The target values. (n_timesteps,)
    forecast: np.array
        The forecast values. (n_samples, n_timesteps)
    region_of_interest: None, int, list of ints, slice, or boolean mask
        The region of interest to apply the roi_metric to.
    roi_weight: float
        The weight to apply to the region of interest.
    constraint: Constraint, optional
        A constraint whose violation must be checked.
    violation_factor: float, default 5.0
        A multiplicative factor to the violation of the constraint, before sending to the exponential.
    log_barrier: bool, default False
        If set to true, the metric is transformed using log(1 + m).

    Returns:
    --------
    result: dict[str, float]
        A dictionary containing the following entries:
        "metric": the final metric.
        "raw_metric": the metric before the log barrier transformation.
        "scaling": the scaling factor applied to the CRPS and the violations.
        "crps": the weighted CRPS.
        "roi_crps": the CRPS only for the region of interest.
        "non_roi_crps": the CRPS only for the forecast not in the region of interest.
        "violation_mean": the average constraint violation over the samples.
        "violation_crps": the CRPS of the constraint violation.
    """
    target_range = np.max(target) - np.min(target)
    scaling = 1.0 / target_range

    if region_of_interest:
        roi_mask = format_roi_mask(region_of_interest, forecast.shape)
        roi_crps = mean_crps(target=target[roi_mask], samples=forecast[:, roi_mask])
        non_roi_crps = mean_crps(
            target=target[~roi_mask], samples=forecast[:, ~roi_mask]
        )
        crps_value = roi_weight * roi_crps + (1 - roi_weight) * non_roi_crps
    else:
        crps_value = mean_crps(target=target, samples=forecast)
        # Those will only be used in the reporting
        roi_crps = crps_value
        non_roi_crps = crps_value

    if constraint:
        violation_amount = constraint.violation(samples=forecast, scaling=scaling)
        violation_exp = np.exp(violation_factor * violation_amount) - 1
        # The target is set to zero, since we make sure that the ground truth always satisfy the constraints
        # The crps code assume multivariate input, so add a dummy dimension
        violation_crps = crps(target=np.zeros(1), samples=violation_exp[:, None])[0]
    else:
        violation_amount = np.zeros(forecast.shape[0])
        violation_exp = np.zeros(forecast.shape[0])
        violation_crps = 0.0

    raw_metric = scaling * crps_value + violation_crps
    if log_barrier:
        metric = np.log(1 + raw_metric)
    else:
        metric = raw_metric

    return {
        "metric": metric,
        "raw_metric": raw_metric,
        "scaling": scaling,
        "crps": scaling * crps_value,
        "roi_crps": scaling * roi_crps,
        "non_roi_crps": scaling * non_roi_crps,
        "violation_mean": violation_amount.mean(),
        "violation_crps": violation_crps,
    }


###############################################################################
############################# OLD METRIC BELOW ################################
###############################################################################
def region_of_interest_constraint_metric(
    target,
    forecast,
    region_of_interest,
    roi_weight,
    roi_metric=mean_crps,
    constraints=None,
    tolerance_percentage=0.05,
):
    """
    Function to calculate a combined metric that considers a region of interest and constraints.

    Parameters:
    ----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    region_of_interest: None, int, list of ints, slice, or boolean mask
        The region of interest to apply the roi_metric to.
    roi_weight: float
        The weight to apply to the region of interest.
    roi_metric: function, optional
        The metric to apply to the region of interest. Default is mean_crps.
    constraints: dict, optional
        A dictionary containing the constraints. The keys are "min" and "max". Default is None.
    tolerance_percentage: float, optional
        The percentage of the target range to use as a tolerance for the constraints. Default is 0.05.

    Returns:
    --------
    combined_roi_metric_with_penalty: float
        The combined metric value with the constraint penalty applied.
    """

    combined_roi_metric = calculate_combined_roi_metric(
        target, forecast, region_of_interest, roi_weight, roi_metric
    )
    if constraints is not None:
        constraint_penalty = calculate_constraint_penalty(
            target, forecast, constraints, tolerance_percentage
        )
    else:
        constraint_penalty = 0

    combined_roi_metric_with_penalty = combined_roi_metric * np.exp(constraint_penalty)

    return combined_roi_metric_with_penalty


def calculate_combined_roi_metric(
    target,
    forecast,
    region_of_interest,
    roi_weight,
    roi_metric,
    divide_by_target_range=True,
):
    """
    Calculate the combined metric for the region of interest and its complement.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    region_of_interest: int, list of ints, slice, or boolean mask
        The region of interest to apply the roi_metric to.
    roi_weight: float
        The weight to apply to the region of interest.
    roi_metric: function
        The metric to apply to the region of interest.
    divide_by_target_range: bool, optional
        Whether to divide the combined metric by the target range. Default is True.

    Returns:
    --------
    combined_roi_metric: float
        The combined metric value for the region of interest and its complement.
    """
    roi_mask = format_roi_mask(region_of_interest, forecast.shape)

    weighted_roi_metric = calculate_masked_metric(
        target, forecast, roi_mask, roi_weight, roi_metric
    )

    weighted_complement_metric = calculate_masked_metric(
        target, forecast, ~roi_mask, (1 - roi_weight), roi_metric
    )
    combined_roi_metric = weighted_roi_metric + weighted_complement_metric

    if divide_by_target_range:
        target_range = np.max(target) - np.min(target)
        return combined_roi_metric / target_range
    else:
        return combined_roi_metric


def calculate_masked_metric(target, forecast, roi_mask, roi_weight, roi_metric):
    """
    Calculate the weighted metric for a masked region of interest.
    Parameters:
    -----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    roi_mask: np.ndarray
        The boolean mask for the region of interest.
    roi_weight: float
        The weight to apply to the region of interest.
    roi_metric: function
        The metric to apply to the region of interest.
    Returns:
    --------
    weighted_roi_metric: float
        The weighted metric value for the region of interest.

    """
    roi_target = target[roi_mask]
    roi_forecast = forecast[:, roi_mask]
    roi_metric = roi_metric(target=roi_target, samples=roi_forecast)
    weighted_roi_metric = roi_weight * roi_metric
    return weighted_roi_metric


def calculate_constraint_penalty(
    target, forecast, constraints, tolerance_percentage, scale_factor=np.log(2)
):
    """
    Calculate the penalty for violating constraints.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    constraints: dict
        A dictionary containing the constraints. The keys are "min" and "max".
    tolerance_percentage: float
        The percentage of the target range to use as a tolerance for the constraints.
    scale_factor: float, optional
        The scale factor to apply to the penalty. Default is np.log(2).

    Returns:
    --------
    penalty: float
        The penalty value for violating constraints.
    """
    penalties = []
    constraint_tolerance = (np.max(target) - np.min(target)) * tolerance_percentage

    for sample_forecast in forecast:
        penalty = 0
        if "min" in constraints:
            min_val = constraints["min"]
            penalty += np.sum(
                np.maximum(0, (min_val - constraint_tolerance) - sample_forecast)
            ) / (np.max(target) - np.min(target))

        if "max" in constraints:
            max_val = constraints["max"]
            penalty += np.sum(
                np.maximum(0, sample_forecast - (max_val + constraint_tolerance))
            ) / (np.max(target) - np.min(target))

        penalty = penalty / len(target)

        penalties.append(penalty)

    average_penalty = np.mean(penalties)
    return average_penalty * scale_factor


###############################################################################
############################# OLD METRIC ABOVE ################################
###############################################################################


def format_roi_mask(region_of_interest, forecast_shape):
    """
    Formats the region of interest mask based on the given inputs.

    Parameters:
    ----------
    region_of_interest (int, list, slice, np.ndarray): The region of interest mask.
        - If an int, a mask with a single True value at the specified index is returned.
        - If a list of ints, a mask with True values at the specified indices is returned.
        - If a slice, a mask with True values within the specified range is returned.
        - If a boolean np.ndarray, the mask is returned as is.
    forecast_shape (tuple): The shape of the forecast.

    Returns:
    -------
        np.ndarray: The formatted region of interest mask.

    """
    if region_of_interest is None:
        mask = np.ones(forecast_shape[1], dtype=bool)
        return mask
    elif isinstance(region_of_interest, int):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        mask[region_of_interest] = True
        return mask
    elif isinstance(region_of_interest, list) and all(
        isinstance(i, int) for i in region_of_interest
    ):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        for i in region_of_interest:
            mask[i] = True
        return mask
    elif isinstance(region_of_interest, slice):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        mask[region_of_interest] = True
        return mask
    elif (
        isinstance(region_of_interest, np.ndarray) and region_of_interest.dtype == bool
    ):
        if region_of_interest.shape[0] != forecast_shape[1]:
            raise ValueError(
                "region_of_interest mask length must match the number of timesteps in the forecast"
            )
        return region_of_interest
    else:
        raise ValueError(
            "region_of_interest must be an int, a list of ints, a slice, or a boolean mask"
        )
