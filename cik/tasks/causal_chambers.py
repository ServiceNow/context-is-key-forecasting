import numpy as np
import pandas as pd

from abc import abstractmethod
from causalchamber.datasets import Dataset
from collections import namedtuple

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..metrics.constraints import MinConstraint, MaxConstraint, ListConstraint
from . import WeightCluster

Window = namedtuple("Window", ["seed", "history_start", "future_start", "time_end"])


class WindTunnelTask(UnivariateCRPSTask):
    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov"]
    __version__ = "0.0.4"  # Modification will trigger re-caching

    def __init__(
        self,
        target_name: str,
        covariate_name: str = "load_in",
        seed: int = None,
        fixed_config: dict = None,
        dataset_name: str = "wt_changepoints_v1",
        datadir: str = DATA_STORAGE_PATH,
    ):
        self.dataset = Dataset(dataset_name, root=datadir, download=True)
        self.seed = seed
        self.covariate_name = covariate_name
        self.target_name = target_name

        super().__init__(seed=seed, fixed_config=fixed_config)

    def _get_number_instances(self):
        """
        Returns number of different instances/windows this task comprises
        """
        return len(self.possible_windows)

    def _get_instance_by_idx(self, idx: int, downsample: str = None):
        """
        Returns instance corresponding to specified index, downsampled if required.

        Parameters
        ----------
        idx : int
            Instance index.
        downsample : str, optional
            Downsampling rule (see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html), by default None

        """

        window = self.possible_windows[idx]

        experiment = self.dataset.get_experiment(name=f"load_in_seed_{window.seed}")
        observations = experiment.as_pandas_dataframe()

        if self.target_name == "pressure_gap":
            observations["pressure_gap"] = (
                observations["pressure_downwind"] - observations["pressure_ambient"]
            )

        selected_variables = observations[
            [self.covariate_name, self.target_name]
        ].copy()
        selected_variables.index = pd.to_datetime(observations.timestamp, unit="s")

        # select past and future
        past_time = selected_variables[window.history_start : window.future_start]
        future_time = selected_variables[window.future_start : window.time_end]

        # get verbalized covariates
        text_covariates = self.verbalize_covariate(
            observations.iloc[window.history_start : window.time_end]
        )

        if downsample is not None:
            # downsample numerical variates, not averaging to avoid introducing new values
            past_time = past_time.resample(downsample).min().ffill()
            future_time = future_time.resample(downsample).min().ffill()

            # A hack to avoid overlapping past and future timestamps introduced by the resampling
            # (this can happen if window.future_start happens in the middle of the new time interval)
            if past_time.index[-1] == future_time.index[0]:
                future_time = future_time[1:]

        return window, past_time, future_time, text_covariates

    @abstractmethod
    def _interval_descriptions(self, interval_start, interval_end):
        pass

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # Not enough history for a single period
        return -1

    def random_instance(self, downsample: str = "1s"):
        """
        Sets random downsampled instance/window as task instance.

        Parameters
        ----------
            downsample : str, optional
                Downsampling rule (see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html), by default "1s".
        """

        # with random choice we are not sure to sample different windows at evaluation
        if self.seed is not None:
            window_idx = self.seed % self._get_number_instances()
        else:
            window_idx = self.random.choice(self._get_number_instances())
        (
            self.window,
            self.past_time,
            self.future_time,
            self.scenario,
        ) = self._get_instance_by_idx(window_idx, downsample)

    def verbalize_covariate(self, observations: pd.DataFrame, round_freq: str = "s"):
        """
        Verbalizes the numerical covariate given time series where change points are marked in the intervention variate.

        Parameters
        ----------
        observations : pd.DataFrame
            Time series that contains the covariate and intervention variate over time
        round_freq : str, optional
            Frequency to which rounding time to, to avoid too long strings, by default "s"

        Returns
        -------
        str
            verbalized covariate
        """

        # intervention column = 0 when load is constant and = 1 when it changes wrt previous timestep
        change_points = list(observations[observations.intervention == 1].index)

        # timestep 0 is always tagged as change point, but we don't need it
        # timestep 0 is not present when window starts at a later timestep
        if 0 in change_points:
            change_points.remove(0)

        # get datetimes from unix times, drop date as it is constant
        timestamps = (
            pd.to_datetime(observations.timestamp, unit="s")
            .dt.round(freq=round_freq)
            .dt.time
        )
        covariate = observations[self.covariate_name]

        return self._interval_descriptions(covariate, change_points, timestamps)


class SpeedFromLoadTask(WindTunnelTask):
    _context_sources = WindTunnelTask._context_sources + ["c_causal", "c_i"]
    _skills = WindTunnelTask._skills + [
        "reasoning: causal",
        "reasoning: math",
        "instruction following",
    ]
    __version__ = "0.0.4"  # Modification will trigger re-caching

    def __init__(
        self,
        seed: int = None,
        fixed_config: dict = None,
        datadir: str = DATA_STORAGE_PATH,
    ):
        self.possible_windows = [
            Window(4, 0, 952, 1100),
            Window(7, 0, 613, 1000),
            Window(3, 300, 807, 1420),
            Window(4, 0, 1886, 2000),
            Window(5, 0, 502, 600),
            Window(6, 0, 686, 880),
            Window(2, 0, 440, 700),
            Window(0, 0, 1159, 1300),
            Window(1, 0, 779, 900),
            Window(1, 0, 779, 1400),
        ]

        super().__init__(
            "rpm_in", "load_in", seed, fixed_config, "wt_changepoints_v1", datadir
        )

        self.background = "The wind tunnel is a chamber with one controllable fan that pushes air through it. We can control the load of the fan (corresponding to the duty cycle of the pulse-width-modulation signal) and measure its speed (in revolutions per minute). The fan is designed so its steady-state speed scales broadly linearly with the load. Unless completely powered off, the fan never operates below a certain speed, corresponding to a minimum effective load between 0.1 and 0.2. The task is to forecast the speed of the fan."
        self.constraints = "The load is between 0 and 1. At full load (=1), the fan turns at a maximum speed of 3000 rpm."
        self.metric_constraint = ListConstraint([MinConstraint(0), MaxConstraint(3000)])

    def _interval_descriptions(self, covariate, change_points, timestamps):
        ans = f"The load is set to: {covariate.iloc[0]:.1f}"
        for c in change_points:
            ans += f" until {timestamps[c]}, {covariate[c]:.1f} from {timestamps[c]}"

        ans += f" until {timestamps.iloc[-1]}."

        return ans


class ExplicitPressureFromSpeedTask(WindTunnelTask):
    _context_sources = WindTunnelTask._context_sources + ["c_causal", "c_i"]
    _skills = WindTunnelTask._skills + [
        "reasoning: causal",
        "reasoning: math",
        "instruction following",
    ]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(
        self,
        seed: int = None,
        fixed_config: dict = None,
        datadir: str = DATA_STORAGE_PATH,
    ):
        self.possible_windows = [
            Window(1, 0, 325, 500),
            Window(1, 200, 550, 650),
            Window(0, 100, 699, 829),
            Window(2, 0, 578, 884),
            Window(3, 700, 1364, 1587),
            Window(4, 600, 952, 1390),
            Window(5, 200, 671, 994),
        ]

        super().__init__(
            "pressure_gap", "rpm_in", seed, fixed_config, "wt_changepoints_v1", datadir
        )

        self.background = "The wind tunnel is a chamber with one controllable fan that pushes air through it. We can control the speed of the fan (rpm_in) and measure the gap between the internal pressure and the ambient pressure (in Pascals). The pressure gap can be estimated from the speed using the affinity laws, which state that the pressure over maximal pressure ratio is proportional to the square of the speed over maximal speed ratio. The task is to forecast the pressure."
        self.constraints = (
            "The maximal fan speed is 3000 rpm and the maximal pressure is 37.5 Pa."
        )
        self.metric_constraint = MaxConstraint(37.5)

    def _interval_descriptions(self, covariate, change_points, timestamps):
        ans = f"The speed starts at {covariate.iloc[0]:.1f}."
        for i, c in enumerate(change_points[:-1]):
            ans += f" At {timestamps[c]}, it rapidly and smoothly changes to {covariate[change_points[i+1]-1]:.1f}."

        ans += f" At {timestamps[change_points[-1]]}, it rapidly and smoothly changes to {covariate.iloc[-1]:.1f}."

        return ans


class ImplicitPressureFromSpeedTask(ExplicitPressureFromSpeedTask):
    _context_sources = WindTunnelTask._context_sources + ["c_causal", "c_i"]
    _skills = WindTunnelTask._skills + [
        "reasoning: causal",
        "retrieval: memory",
        "reasoning: math",
        "instruction following",
    ]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(
        self,
        seed: int = None,
        fixed_config: dict = None,
        datadir: str = DATA_STORAGE_PATH,
    ):
        super().__init__(seed, fixed_config, datadir)

        self.background = "The wind tunnel is a chamber with one controllable fan that pushes air through it. We can control the speed of the fan (in revolutions per minute) and measure the gap between the internal pressure and the ambient pressure (in Pascals). The pressure gap can be estimated from the speed using the affinity laws. The task is to forecast the pressure."


__TASKS__ = [
    SpeedFromLoadTask,
    ExplicitPressureFromSpeedTask,
    ImplicitPressureFromSpeedTask,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            SpeedFromLoadTask,
        ],
    ),
    WeightCluster(
        weight=1,
        tasks=[
            ExplicitPressureFromSpeedTask,
            ImplicitPressureFromSpeedTask,
        ],
    ),
]
