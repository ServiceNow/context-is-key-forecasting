import numpy as np

from .metrics.constraints import MinConstraint, MaxConstraint, VariableMaxConstraint


def add_realistic_noise(
    series,
    random,
    noise_level=0.03,
    skip_zero_values=False,
    constraints=[],
):
    """
    Adds Gaussian noise to the series. The standard deviation of the noise
    is a fraction of the series' standard deviation.

    Parameters:
    - series: pandas Series, the original time series data
    - random: random state
    - noise_level: float, the fraction of the series' std to use as noise std
    - skip_zero_values: bool, can be enabled to avoid adding noise to zero values in the series; this can be used if zero has an important meaning in the series
    - constraints: list, list of constraints

    Returns:
    - pandas Series with added noise
    """
    # Use the provided random_state or np.random as default
    rng = random
    # Calculate the standard deviation of the series
    series_std = series.std()
    # Determine the standard deviation of the noise
    noise_std = noise_level * series_std
    # Generate Gaussian noise
    noise = rng.normal(0, noise_std, size=series.shape)
    # If skip_zeros is True, set noise to zero where the original series has zeros
    if skip_zero_values:
        noise[series == 0] = 0
    # Add noise to the original series
    noisy_series = series + noise
    # Apply constraints
    for constraint in constraints:
        if type(constraint) == MinConstraint:
            noisy_series[noisy_series < constraint.threshold] = constraint.threshold
        elif type(constraint) == MaxConstraint:
            noisy_series[noisy_series > constraint.threshold] = constraint.threshold
        elif type(constraint) == VariableMaxConstraint:
            # Apply the constraint at each specified index
            for idx, threshold in zip(constraint.indices, constraint.thresholds):
                # Ensure the index is within the bounds of the data
                if idx >= 0 and idx < len(noisy_series):
                    if noisy_series.iloc[idx] > threshold:
                        noisy_series.iloc[idx] = threshold
    return noisy_series


def log_transform(series):
    """
    Applies natural logarithm to the series, handling zeros appropriately.

    Parameters:
    - series: pandas Series, the original time series data

    Returns:
    - pandas Series after log transform
    """
    # Raise error if negative numbers exist
    if (series < 0).any():
        negative_values = series[series < 0]
        raise ValueError(
            f"Cannot apply log transform. Negative values detected in the series:\n{negative_values}"
        )
    # Replace zeros with a small positive value to avoid log(0)
    series = series.replace(0, 1e-6)
    # Apply the natural logarithm
    log_series = np.log(series)
    return log_series
