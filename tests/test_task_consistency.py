"""
Unit tests that check if tasks are consistent with our standards

"""

import pandas as pd
import pytest

from benchmark import ALL_TASKS, WEIGHT_CLUSTERS
from benchmark.base import BaseTask, ALLOWED_CONTEXT_SOURCES, ALLOWED_SKILLS
from benchmark.utils import get_all_parent_classes


@pytest.mark.parametrize("task", ALL_TASKS)
def test_inherits_base(task):
    """
    Test that each task inherits the base class

    """
    assert issubclass(task, BaseTask)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_time_data_is_dataframe(task):
    """
    Test that the temporal data is given as a Pandas dataframe

    """
    task_instance = task()
    assert isinstance(task_instance.past_time, pd.DataFrame)
    assert isinstance(task_instance.future_time, pd.DataFrame)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_past_time_is_long_enough(task):
    """
    Test that the historical data is long enough for the ExponentialSmoothingForecaster minimum requirement

    """
    task_instance = task()
    assert len(task_instance.past_time) >= 3


@pytest.mark.parametrize("task", ALL_TASKS)
def test_some_context_exists(task):
    """
    Test that at least part of the context is non-empty

    """
    task_instance = task()
    assert (
        task_instance.background or task_instance.constraints or task_instance.scenario
    )


@pytest.mark.parametrize("task", ALL_TASKS)
def test_context_sources(task):
    """
    Test the task marks at least one kind of context as being used and that all listed context sources are allowed.

    """
    assert len(task._context_sources) > 0
    assert all(ctx in ALLOWED_CONTEXT_SOURCES for ctx in task._context_sources)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_skills(task):
    """
    Test the task marks at least three kinds of skill as being used and that all listed skills are allowed.

    """
    assert len(task._skills) > 2
    assert all(skill in ALLOWED_SKILLS for skill in task._skills)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_version(task):
    """
    Test that the task defines a version attribute

    """
    assert (
        "__version__" in task.__dict__
    ), f"{task} should define a __version__ attribute"
    parents = get_all_parent_classes(task)
    status = {t: "__version__" in t.__dict__ for t in parents}
    assert all(
        status.values()
    ), f"All parents of {task} should define a __version__ attribute but found {status}"


def test_unique_cluster():
    """
    Test that each task is in exactly one weight cluster
    """
    for task in ALL_TASKS:
        found = 0
        for cluster in WEIGHT_CLUSTERS:
            if task in cluster.tasks:
                found += 1
        assert found == 1, f"{task} must be in exactly one weight cluster"


def test_no_extra_task_in_cluster():
    """
    Test that each task which is in a cluster, is also in ALL_TASKS
    """
    for cluster in WEIGHT_CLUSTERS:
        for task in cluster.tasks:
            assert (
                task in ALL_TASKS
            ), f"{task} is in a weight cluster, so it must be in ALL_TASKS"
