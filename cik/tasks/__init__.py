from dataclasses import dataclass
from fractions import Fraction
from typing import Union
from ..base import BaseTask


@dataclass
class WeightCluster:
    """Group of tasks which splits their weight amongst each other"""

    weight: Union[int, Fraction]
    tasks: list[BaseTask]
