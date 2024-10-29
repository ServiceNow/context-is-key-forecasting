"""
Unit tests for causal chambers tasks

"""

import pandas as pd

from benchmark.tasks.causal_chambers import SpeedFromLoadTask


def test_deterministic_instance():
    """
    Tests that two instances are loaded correctly: time series have the right history and prediction lengths, have exactly two variates and are verbalized correctly.
    """

    task = SpeedFromLoadTask()
    assert task._get_number_instances() == 10

    idx = 0
    window, past_time, future_time, covariates = task._get_instance_by_idx(idx)

    assert len(past_time.index) == window.future_start - window.history_start
    assert len(future_time.index) == window.time_end - window.future_start

    assert len(past_time.columns) == 2
    assert len(future_time.columns) == 2

    assert covariates.count("until") == 5
    assert covariates.count("from") == 4

    idx = 2
    window, past_time, future_time, covariates = task._get_instance_by_idx(idx)

    assert len(past_time.index) == window.future_start - window.history_start
    assert len(future_time.index) == window.time_end - window.future_start

    assert len(past_time.columns) == 2
    assert len(future_time.columns) == 2

    assert covariates.count("until") == 6
    assert covariates.count("from") == 5


def test_downsampling():
    """
    Tests random instances are loaded and downsampled correctly (keeping 1 over 6-7 original timesteps)
    """

    for _ in range(50):
        task = SpeedFromLoadTask()
        task.random_instance()

        # downsampling to 1s, while original frequency should be ~7 Hertz
        assert (
            len(task.past_time.index)
            < (task.window.future_start - task.window.history_start) / 6
        )
        assert (
            len(task.future_time.index)
            < (task.window.time_end - task.window.future_start) / 6
        )
