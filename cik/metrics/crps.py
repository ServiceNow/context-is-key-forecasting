# Taken from https://github.com/ServiceNow/regions-of-reliability/blob/main/ror/metrics/crps.py
"""
Copyright 2023 ServiceNow
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
