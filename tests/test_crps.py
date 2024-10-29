"""
Unit test for the CRPS metric.
"""

import numpy as np
import pytest

from benchmark.metrics.crps import crps


def crps_slow(target: np.array, samples: np.array) -> np.float32:
    # Reference implementation, which is easy to understand as being correct.
    # Compute the CRPS using the definition:
    # CRPS(y, X) = E |y - X| + 0.5 E |X - X'|, averaged over each dimension
    assert target.shape[0] == samples.shape[1]
    num_samples = samples.shape[0]

    first_term = np.abs(samples - target[None, :]).mean(axis=0)
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            s += np.abs(samples[i] - samples[j])
    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term


@pytest.mark.parametrize("seed", range(1, 6))
def test_same_as_crps_slow(seed):
    """
    Test that new implementation gives the exact same result as the reference implementation,
    which is easier to read and understand.
    """
    random = np.random.RandomState(seed)

    target = random.normal(0, 1, size=10)
    samples = random.normal(0, 1, size=(25, 10))

    reference = crps_slow(target, samples)
    test_version = crps(target, samples)

    assert np.isclose(reference, test_version, rtol=1e-05, atol=1e-08).all()


@pytest.mark.parametrize("seed", range(1, 6))
def test_variable_dimensions(seed):
    """
    Test that the CRPS works with arbitrary number of dimensions, and gives consistent results.
    """
    random = np.random.RandomState(seed)

    target_3d = random.normal(0, 1, size=(2, 3, 4))
    samples_3d = random.normal(0, 1, size=(15, 2, 3, 4))
    target_2d = target_3d.reshape(2, 3 * 4)
    samples_2d = samples_3d.reshape(15, 2, 3 * 4)
    target_1d = target_3d.reshape(2 * 3 * 4)
    samples_1d = samples_3d.reshape(15, 2 * 3 * 4)

    crps_3d = crps(target_3d, samples_3d)
    crps_2d = crps(target_2d, samples_2d)
    crps_1d = crps(target_1d, samples_1d)

    assert np.isclose(crps_3d.reshape(2 * 3 * 4), crps_1d, rtol=1e-05, atol=1e-08).all()
    assert np.isclose(crps_2d.reshape(2 * 3 * 4), crps_1d, rtol=1e-05, atol=1e-08).all()
