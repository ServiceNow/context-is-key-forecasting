"""
Utility functions

"""

import numpy as np
import pandas as pd
from typing import Union


def get_random_window_univar(
    series,
    prediction_length,
    history_factor: int = 1,
    random: np.random.RandomState = np.random,
) -> pd.Series:
    """
    Get a random window of a given size from a univariate time series

    Parameters
    ----------
    series : pandas.Series
        Time series
    prediction_length : int
        The length of the portion of the window to forecast
    history_factor : int, optional, default = 1
        Factor by which the history length is larger than the prediction length
    random: numpy.random.RandomState, optional
        Random number generator

    Returns
    -------
    pandas.Series
        Random window of the given size

    """
    history_length = min(
        history_factor * prediction_length, len(series) - prediction_length
    )

    # Random window selection that ensures sufficient history
    window_start = random.randint(0, len(series) - history_length - prediction_length)
    window_end = window_start + history_length + prediction_length

    return series.iloc[window_start:window_end]


def datetime_to_str(dt: Union[pd.Timestamp, np.datetime64]) -> str:
    """
    Convert multiple possible representation of the same datetime into the same output.

    Parameters
    ----------
    dt : pandas.Timestamp or numpy.datetime64
        The datetime to be converted

    Returns
    -------
    str
        The datetime in the YYYY-MM-DD HH:MM:SS format
    """
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
