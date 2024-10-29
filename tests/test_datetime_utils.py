"""
Unit tests that check if the various utils converning date time manipulation are correct.
"""

import pandas as pd
import numpy as np
import pytest

from benchmark.utils import datetime_to_str


@pytest.mark.parametrize(
    "dt,result",
    [
        (pd.Timestamp("2017-01-01T12"), "2017-01-01 12:00:00"),
        (pd.Timestamp(2016, 4, 5, 16), "2016-04-05 16:00:00"),
        (pd.Timestamp(2016, 4, 5, 16, 30, 17), "2016-04-05 16:30:17"),
        (
            pd.Timestamp(
                year=2016,
                month=4,
                day=5,
                hour=16,
                minute=30,
                second=17,
                microsecond=500,
            ),
            "2016-04-05 16:30:17",
        ),
    ],
)
def test_timestamp_to_str(dt: pd.Timestamp, result: str):
    """
    Test that we get the expected output with panda.Timestamp objects.
    """
    assert datetime_to_str(dt) == result


@pytest.mark.parametrize(
    "dt,result",
    [
        (np.datetime64("2017-01-01T12"), "2017-01-01 12:00:00"),
        (np.datetime64("2017-01-01T12", "Y"), "2017-01-01 00:00:00"),
        (np.datetime64("2017-01-01T12:00:00"), "2017-01-01 12:00:00"),
        (np.datetime64("2017-01-01T12:30:17"), "2017-01-01 12:30:17"),
        (np.datetime64("2017-01-01T12:30:17.342343"), "2017-01-01 12:30:17"),
    ],
)
def test_datetime64_to_str(dt: np.datetime64, result: str):
    """
    Test that we get the expected output with np.datetime64 objects.
    """
    assert datetime_to_str(dt) == result
