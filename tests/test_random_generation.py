import pytest

from benchmark import ALL_TASKS


def _are_instances_equal(instance_1, instance_2):
    """
    Checks if two task instances are identical

    """
    # Check if temporal data is the same across instances
    same_history = instance_1.past_time.equals(instance_2.past_time)
    same_future = instance_1.future_time.equals(instance_2.future_time)
    same_data = same_history and same_future

    # Check if the background, constraints, and scenario are the same
    same_background = instance_1.background == instance_2.background
    same_constraints = instance_1.constraints == instance_2.constraints
    same_scenario = instance_1.scenario == instance_2.scenario

    # Confirm that at least something is different
    return same_data and same_background and same_constraints and same_scenario


@pytest.mark.parametrize("task", ALL_TASKS)
def test_instance_randomness(task):
    """
    Test that each task can produce random instances

    """
    for seed in range(5):
        assert not _are_instances_equal(task(seed=seed), task(seed=seed + 1))


@pytest.mark.parametrize("task", ALL_TASKS)
def test_seed_consistency(task):
    """
    Test that calling the constructor with the same seed leeds to the
    same data all the time

    """
    for seed in range(5):
        assert _are_instances_equal(task(seed=seed), task(seed=seed))
