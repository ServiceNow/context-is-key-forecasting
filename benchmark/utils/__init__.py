"""
Utility functions

"""

import abc
import builtins
import inspect
import logging
import numpy as np
import pandas as pd
from typing import Union


def get_random_window_univar(
    series,
    prediction_length,
    history_factor: int = 1,
    avoid_nulls: bool = True,
    random: np.random.RandomState = np.random,
    max_attempts: int = 100,
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
    avoid_nulls : bool, optional, default = True
        Whether to avoid series with a history or future where all values are
        close to zero.
    random: numpy.random.RandomState, optional
        Random number generator
    max_attempts: int, optional, default = 100
        Maximum number of attempts to find a valid window. Raise an error if
        this number is exceeded.

    Returns
    -------
    pandas.Series
        Random window of the given size

    """
    for _ in range(max_attempts):
        history_length = min(
            history_factor * prediction_length, len(series) - prediction_length
        )

        # Random window selection that ensures sufficient history
        window_start = random.randint(
            0, len(series) - history_length - prediction_length
        )
        window_end = window_start + history_length + prediction_length

        window = series.iloc[window_start:window_end]
        history = window.iloc[:history_length]
        future = window.iloc[history_length:]

        if avoid_nulls:
            if np.allclose(history, 0) or np.allclose(future, 0):
                logging.debug("Randomly selected window rejected due to nulls.")
                continue  # Reject window

        logging.debug("Randomly selected window accepted.")
        break  # No criteria violated, accept window

    else:
        raise ValueError(f"Could not find a valid window after {max_attempts} attempts")

    return window


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


def get_all_parent_classes(cls):
    """
    Get all parent classes of a class, excluding built-in classes and the ABC class.

    Parameters
    ----------
    cls : class
        The class to get the parent classes of.

    Returns:
    --------
    list
        List of parent classes
    """
    return [
        c
        for c in inspect.getmro(cls)
        if c.__module__ != builtins.__name__
        if c != cls
        if c is not abc.ABC
    ]
