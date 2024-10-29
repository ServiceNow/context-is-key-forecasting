"""
Compute the scaling coefficient for the metrics on all tasks.
This must be run to allow UnivariateCRPSTask subclasses to compute their metrics.
"""

import traceback
import logging

from benchmark.metrics.scaling_cache import ScalingCache, DefaultScalingCache
from benchmark import ALL_TASKS


logging.basicConfig(level=logging.INFO)


ComputeScalingCache = ScalingCache(
    scaling_method=DefaultScalingCache.scaling_method,
    seeds=DefaultScalingCache.seeds,
    raise_on_miss=False,
    compute_on_miss=True,
)

for task_cls in ALL_TASKS:
    try:
        _ = ComputeScalingCache(task_class=task_cls)
    except Exception as e:
        print(f"Error computing scaling for task {task_cls.__name__}")
        print(str(e))
        print(traceback.format_exc())
