import builtins
import inspect
import hashlib
import logging
import pandas as pd

from pathlib import Path


from .disk_cache import HDF5DiskCache
from ...baselines.base import Baseline
from ...config import DEFAULT_N_SAMPLES, RESULT_CACHE_PATH
from ...utils import get_all_parent_classes


def get_method_cache_name(method_callable):
    """
    Convert a method to a string that can be used as a cache path.

    Parameters:
    -----------
    method_callable: callable
        A callable that receives a task instance and a number of samples and returns
        a prediction samples. The callable should expect the following kwargs:
        task_instance, n_samples

    """
    if method_callable.__class__.__name__ == "function":
        return f"{method_callable.__module__}.{method_callable.__qualname__}"
    elif isinstance(method_callable, Baseline):
        return method_callable.cache_name
    else:
        raise ValueError("Unable to infer cache name for method.")


def get_source(obj) -> str:
    """
    Get the source code of an object as a string.

    Parameters:
    -----------
    obj: callable
        Any object

    Notes:
    ------
    If the object is a class, the source code of all parent classes is also included.

    """
    if obj.__class__.__name__ == "function":
        return inspect.getsource(obj)
    else:
        # Get all parent classes that are not built-in
        parent_classes = get_all_parent_classes(obj.__class__)
        # Concatenate source code of all parent classes and the method's class
        return "".join(inspect.getsource(c) for c in parent_classes + [obj.__class__])


def get_versions(obj) -> str:
    """
    Get the versions associated with an object as a string.

    Parameters:
    -----------
    obj: callable
        Any object

    Notes:
    ------
    If the object is a class, the version of all parent classes is also included.

    """
    if obj.__class__.__name__ == "function":
        return obj.__version__
    else:
        # Get all parent classes that are not built-in
        parent_classes = get_all_parent_classes(obj.__class__)
        # Concatenate source code of all parent classes and the method's class
        return ",".join(c.__version__ for c in parent_classes + [obj.__class__])


class CacheMissError(Exception):
    pass


class ResultCache:
    """
    A cache to avoid recomputing costly results. Basically acts as a wrapper
    around a method callable that caches the results.

    Parameters:
    -----------
    method_callable: callable
        A callable that receives a task instance and a number of samples and returns
        a prediction samples. The callable should expect the following kwargs:
        task_instance, n_samples
    method_name: str, optional
        Name of the method callable. Used to create a directory in the cache.
        Must be provided if method_callable is an instance of a class.
    cache_path: str, optional
        Path to the cache directory. Default is taken from RESULT_CACHE.
    cache_method: str, optional
        Method to use for caching. Default is "versions", which will look at the version of the
        task class and method callable (and all parent classes, if applicable) to create a cache key.
        Other option is "code", which uses the source code of the method and the task to create the key.
    raise_on_miss: bool, optional
        Whether to raise a CacheMissError if the cache is not found. Default is False.

    """

    def __init__(
        self,
        method_callable,
        method_name=None,
        cache_path=RESULT_CACHE_PATH,
        cache_method="versions",
        raise_on_miss=False,
    ) -> None:
        self.logger = logging.getLogger("Result cache")
        self.method_callable = method_callable
        self.raise_on_miss = raise_on_miss

        # Cache configuration
        self.cache_dir = Path(cache_path) / (
            get_method_cache_name(method_callable)
            if method_name is None
            else method_name
        )
        self.cache = HDF5DiskCache(self.cache_dir)

        # Set the cache key calculation method
        self.cache_method = cache_method
        if self.cache_method == "code":
            self.key_extractor = get_source
        elif self.cache_method == "versions":
            self.key_extractor = get_versions
        else:
            raise ValueError("Invalid cache method.")

    def get_cache_key(self, task_instance, n_samples):
        """
        Get cache key by hashing the task instance (including all data and code)
        as well as the method callable's source code.

        Any change in the task instance or the method callable will result in recomputation.

        """

        def get_attr_hash(attr):
            if isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series):
                # Convert DataFrame to string representation of its data
                return attr.to_csv().encode("utf-8")
            elif isinstance(attr, str):
                # Directly encode strings
                return attr.encode("utf-8")
            else:
                # Convert other attribute types to string and encode
                return str(attr).encode("utf-8")

        # Initialize the hash object
        hasher = hashlib.sha256()

        # Hash the task
        hasher.update(self.key_extractor(task_instance).encode("utf-8"))

        # Hash the method
        hasher.update(self.key_extractor(self.method_callable).encode("utf-8"))

        # Hash the number of samples
        hasher.update(str(n_samples).encode("utf-8"))

        # Hash the task attributes that get sent to the method callable
        hasher.update(get_attr_hash(task_instance.past_time))
        hasher.update(get_attr_hash(task_instance.future_time))
        hasher.update(get_attr_hash(task_instance.background))
        hasher.update(get_attr_hash(task_instance.constraints))
        hasher.update(get_attr_hash(task_instance.scenario))

        # Return the hex digest of the hash
        return hasher.hexdigest()

    def __call__(self, task_instance, n_samples=DEFAULT_N_SAMPLES):
        self.logger.info("Attempting to load from cache.")
        cache_key = self.get_cache_key(task_instance, n_samples)

        # Attempt to get from cache
        samples = self.cache.get(cache_key, None)
        if samples is not None:
            self.logger.info("Cache hit.")
            return samples

        if self.raise_on_miss:
            raise CacheMissError()

        self.logger.info("Cache miss. Running inference.")
        samples = self.method_callable(task_instance, n_samples)

        # Update cache on disk
        self.logger.info("Updating cache.")
        self.cache[cache_key] = samples

        return samples

    def __str__(self) -> str:
        return str(self.method_callable)
