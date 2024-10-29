__version__ = "0.0.1"

from fractions import Fraction

from .base import BaseTask

from .tasks.electricity_tasks import (
    __TASKS__ as ELECTRICITY_TASKS,
    __CLUSTERS__ as ELECTRICITY_CLUSTERS,
)
from .tasks.nn5_tasks import __TASKS__ as NN5_TASKS, __CLUSTERS__ as NN5_CLUSTERS
from .tasks.pred_change_tasks import (
    __TASKS__ as PRED_CHANGE_TASKS,
    __CLUSTERS__ as PRED_CHANGE_CLUSTERS,
)
from .tasks.predictable_constraints_real_data import (
    __TASKS__ as PREDICTABLE_CONSTRAINT_TASKS,
    __CLUSTERS__ as PREDICTABLE_CONSTRAINT_CLUSTERS,
)
from .tasks.predictable_grocer_shocks import (
    __TASKS__ as PREDICTABLE_GROCER_SHOCKS_TASKS,
    __CLUSTERS__ as PREDICTABLE_GROCER_SHOCKS_CLUSTERS,
)

# from .tasks.predictable_spikes_in_pred import (
#     __TASKS__ as PREDICTABLE_SPIKES_IN_PRED_TASKS,
# )
# from .tasks.predictable_stl_shocks import __TASKS__ as PREDICTABLE_STL_SHOCKS_TASKS
from .tasks.sensor_maintenance import (
    __TASKS__ as SENSOR_MAINTENANCE_TASKS,
    __CLUSTERS__ as SENSOR_MAINTENANCE_CLUSTERS,
)
from .tasks.causal_chambers import (
    __TASKS__ as CAUSAL_CHAMBERS_TASKS,
    __CLUSTERS__ as CAUSAL_CHAMBERS_CLUSTERS,
)
from .tasks.bivariate_categorical_causal import (
    __TASKS__ as CATEGORICAL_CAUSAL_TASKS,
    __CLUSTERS__ as CATEGORICAL_CAUSAL_CLUSTERS,
)
from .tasks.solar_tasks import __TASKS__ as SOLAR_TASKS, __CLUSTERS__ as SOLAR_CLUSTERS
from .tasks.traffic_tasks import (
    __TASKS__ as TRAFFIC_TASKS,
    __CLUSTERS__ as TRAFFIC_CLUSTERS,
)
from .tasks.nsrdb_tasks import __TASKS__ as NSRDB_TASKS, __CLUSTERS__ as NSRDB_CLUSTERS
from .tasks.montreal_fire import (
    __TASKS__ as MONTREAL_FIRE_TASKS,
    __CLUSTERS__ as MONTERAL_FIRE_CLUSTERS,
)
from .tasks.fred_county_tasks import (
    __TASKS__ as FRED_COUNTY_TASKS,
    __CLUSTERS__ as FRED_COUNTY_CLUSTERS,
)
from .tasks.pems_tasks import __TASKS__ as PEMS_TASKS, __CLUSTERS__ as PEMS_CLUSTERS

# All tasks that are officially included in the benchmark
ALL_TASKS = (
    ELECTRICITY_TASKS
    + NN5_TASKS
    + PRED_CHANGE_TASKS
    + PREDICTABLE_CONSTRAINT_TASKS
    # + PREDICTABLE_GROCER_SHOCKS_TASKS
    + SENSOR_MAINTENANCE_TASKS
    + CAUSAL_CHAMBERS_TASKS
    + CATEGORICAL_CAUSAL_TASKS
    + SOLAR_TASKS
    + TRAFFIC_TASKS
    + NSRDB_TASKS
    + MONTREAL_FIRE_TASKS
    + FRED_COUNTY_TASKS
    # + PEMS_TASKS
)

WEIGHT_CLUSTERS = (
    ELECTRICITY_CLUSTERS
    + NN5_CLUSTERS
    + PRED_CHANGE_CLUSTERS
    + PREDICTABLE_CONSTRAINT_CLUSTERS
    # + PREDICTABLE_GROCER_SHOCKS_CLUSTERS
    + SENSOR_MAINTENANCE_CLUSTERS
    + CAUSAL_CHAMBERS_CLUSTERS
    + CATEGORICAL_CAUSAL_CLUSTERS
    + SOLAR_CLUSTERS
    + TRAFFIC_CLUSTERS
    + NSRDB_CLUSTERS
    + MONTERAL_FIRE_CLUSTERS
    + FRED_COUNTY_CLUSTERS
    # + PEMS_CLUSTERS
)


def get_task_weight(task: BaseTask) -> Fraction:
    for cluster in WEIGHT_CLUSTERS:
        if task in cluster.tasks:
            return Fraction(cluster.weight) / len(cluster.tasks)


TASK_NAME_TO_WEIGHT = {task.__name__: get_task_weight(task) for task in ALL_TASKS}
