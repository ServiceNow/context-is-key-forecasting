import hashlib
import logging

from pathlib import Path
from typing import Optional

from ..config import METRIC_SCALING_CACHE_PATH
from ..utils import get_all_parent_classes
from ..utils.cache.disk_cache import HDF5DiskCache


# Cannot be imported from the benchmark.utils.cache due to circular imports
class CacheMissError(Exception):
    pass


def get_versions_class(cls) -> str:
    """
    Get the versions associated with a class as a string.

    Parameters:
    -----------
    cls: class
        Any class

    Notes:
    ------
    The version of all parent classes is also included.

    """
    # Get all parent classes that are not built-in
    parent_classes = get_all_parent_classes(cls)
    # Concatenate source code of all parent classes and the method's class
    return ",".join(c.__version__ for c in parent_classes + [cls])


class ScalingCache:
    """
    A cache to avoid recomputing the scaling factor for the tasks.

    Parameters:
    -----------
    scaling_method: callable
        A callable that receives a task class and a list of seeds and compute the average
        scaling factor for said task. The callable should expect the following kwargs:
        task_class, seeds
    seeds: list of int
        Which seeds to use to compute the scaling factor
    cache_path: str, optional
        Path to the cache directory. Default is taken from RESULT_CACHE.
    raise_on_miss: bool, optional
        Whether to raise a CacheMissError if the cache is not found. Default is False.
    compute_on_miss: bool, optional
        Whether to compute the scaling factor if the cache is not found.
        If False, the method will return None instead.
        Default is False.

    """

    def __init__(
        self,
        scaling_method,
        seeds,
        cache_path=METRIC_SCALING_CACHE_PATH,
        raise_on_miss=False,
        compute_on_miss=False,
    ) -> None:
        self.logger = logging.getLogger("Metric scaling cache")
        self.scaling_method = scaling_method
        self.seeds = seeds
        self.raise_on_miss = raise_on_miss
        self.compute_on_miss = compute_on_miss

        # Cache configuration
        self.cache_dir = Path(cache_path)
        # Lazy initialization of the cache, to allow loading this Python file where the cache is unavailable
        # ex: when running tests on Github
        self.cache: Optional[HDF5DiskCache] = None

    def get_cache_key(self, task_class, seeds: list[int]):
        """
        Get cache key by hashing the task class version tags + its name, the scaling method version id, and the seeds.
        """

        # Initialize the hash object
        hasher = hashlib.sha256()

        # Hash the task
        hasher.update(self.scaling_method.__version__.encode("utf-8"))
        # Don't use str(task_class) to avoid invalidating the cache if we move the class to a new folder
        hasher.update(task_class.__name__.encode("utf-8"))
        hasher.update(get_versions_class(task_class).encode("utf-8"))
        hasher.update(",".join([str(s) for s in seeds]).encode("utf-8"))

        # Return the hex digest of the hash
        return hasher.hexdigest()

    def __call__(self, task_class) -> Optional[float]:
        if self.cache is None:
            self.logger.info("First call, initializing cache")
            self.cache = HDF5DiskCache(self.cache_dir)

        self.logger.info("Attempting to load from cache.")
        cache_key = self.get_cache_key(task_class, self.seeds)
        if cache_key in self.cache:
            self.logger.info("Cache hit.")
            return self.cache[cache_key]

        if self.raise_on_miss:
            raise CacheMissError()

        if self.compute_on_miss:
            self.logger.info("Cache miss. Computing scaling.")
            scaling = self.scaling_method(task_class=task_class, seeds=self.seeds)

            # Update cache on disk
            self.logger.info("Updating cache.")
            self.cache[cache_key] = scaling

            return scaling
        else:
            self.logger.warning(f"Cache miss for {task_class}.")
            return None


def inverse_mean_forecast_range(task_class, seeds) -> float:
    """
    For each instance, compute the width of possibles values in the forecast window:
    max(ground_truth) - min(ground_truth).
    Then, the scaling will be 1 divided by the average of these widths
    """
    widths = []
    for s in seeds:
        instance = task_class(seed=s)
        only_column = instance.future_time.columns[-1]
        target = instance.future_time[only_column]
        widths.append(target.max() - target.min())
    mean_width = sum(widths) / len(widths)
    return 1 / mean_width


inverse_mean_forecast_range.__version__ = (
    "0.0.1"  # Modification will trigger re-caching
)


# This is the Scaling Cache that is connected to the UnivariateCRPSTask evaluation code
# It currently computes on miss, so it doesn't need to be precomputed (see precompute_scaling_cache.py)
DefaultScalingCache = ScalingCache(
    scaling_method=inverse_mean_forecast_range,
    seeds=list(range(1001, 1026)),
    raise_on_miss=False,
    compute_on_miss=True,
)
