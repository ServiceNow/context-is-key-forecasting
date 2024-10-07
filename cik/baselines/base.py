"""
Base classes for baselines

"""

import numpy as np

from abc import ABC, abstractmethod


class Baseline(ABC):
    """
    Base class for baselines

    """

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, task_instance, n_samples: int) -> np.ndarray:
        """
        Infer forecast for a task instance using the baseline

        """
        pass

    @property
    def cache_name(self) -> str:
        return self.__class__.__name__

    def __str__(self):
        return self.cache_name
