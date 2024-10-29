"""
Copyright 2024 ServiceNow
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

import numpy as np


def crps_quantile(
    target: np.array,
    samples: np.array,
    quantiles: np.array = (np.arange(20) / 20.0)[1:],
) -> np.float32:
    # Compute the CRPS using the quantile scores
    # For a single series
    assert target.shape[0] == samples.shape[1]
    num_samples = samples.shape[0]

    sorted_samples = np.sort(samples, axis=0)

    quantile_scores = {}

    for q in quantiles:
        # From 0 to num_samples - 1 so that the 100% quantile is the last sample
        q_idx = int(np.round(q * (num_samples - 1)))
        q_value = sorted_samples[q_idx, :]

        # The absolute value is just there in case of numerical inaccuracies
        quantile_score = np.abs(2 * ((target <= q_value) - q) * (q_value - target))

        # result_sum += np.nanmean(quantile_score, axis=(0, 1))
        quantile_scores[q] = quantile_score.sum(axis=0)

    abs_target_sum = np.atleast_1d(np.abs(target).sum(axis=0))

    series_QuantileLoss = np.array(
        [quantile_scores[quantile] for quantile in quantiles]
    ).transpose()  # (num_series, num_quantiles)

    series_QuantileLoss = (
        series_QuantileLoss.reshape(1, -1)
        if series_QuantileLoss.ndim == 1
        else series_QuantileLoss
    )
    # if normalize:
    #     result_sum /= np.sum(np.abs(target))

    return series_QuantileLoss, abs_target_sum


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
