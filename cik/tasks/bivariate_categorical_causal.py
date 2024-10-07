from datetime import datetime, timedelta
from ..base import UnivariateCRPSTask
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from termcolor import colored
from abc import abstractmethod
from ..utils.causal import (
    check_dagness,
    parent_descriptions,
    get_historical_parents,
    generate_timestamps,
    truncate_regime,
    verbalize_variable_values,
)
from . import WeightCluster

"""
    Usage:
    task = MinimalCausalContextBivarLinSVAR(seed=1)
    or
    task = FullCausalContextImplicitEquationBivarLinSVAR(seed=1)
    or
    task = FullCausalContextExplicitEquationBivarLinSVAR(seed=1)

    plot_forecast_with_covariates(task, f"{task.plot_name}.png")
"""


class CausalUnivariateCRPSTask(UnivariateCRPSTask):
    """
    Base class for all synthetic causal tasks that require forecasting a single series and that use CRPS for evaluation

    """

    __version__ = "0.0.3"  # Modification will trigger re-caching

    # To be used only for the instantaneous graph
    def generate_dag(self, num_nodes, degree):
        """
        Generate Erdos-Renyi random graph with a given degree

        Required only for making the instantaneous graphs since the lagged graphs
        do not have to be acyclic.
        """

        total_edges = num_nodes * (num_nodes - 1) // 2
        total_expected_edges = degree * num_nodes
        p_threshold = float(total_expected_edges) / total_edges
        p_edge = (self.random.rand(num_nodes, num_nodes) < p_threshold).astype(float)
        L = np.tril(p_edge, k=-1)

        P = self.random.permutation(np.eye(num_nodes, num_nodes))
        G = P.T @ L @ P
        return G

    def random_graph(
        self, num_nodes, intra_degree, inter_degree, lag, instantaneous_edges, split
    ):
        """
        A function to generate a random (instantaneous, lagged) DAG graph for the causal task

        Parameters
        ----------
            num_nodes: int

            intra_degree: float
                Expected edges for instantaneous graph

            inter_degree: float
                Expected edges for lagged graph

            lag: int
                Lag timesteps for the causal

            instantaneous_edges: bool
                If True, the graph will have instantaneous connections and intra_degree will be used
                Else intra_degree is not used for any step of the random DAG generation.

            split: Tuple(List, List)
                Forecast variable list and covariate list

        Returns
        -------
        A complete DAG such that the forecast variable has at least one covariate as a parent.

            graph: np.array
                Binary adjacency matrix of shape (lag+1, d, d)
        """

        assert lag > 0, "Lag must be greater than 0"
        assert num_nodes > 1, "DAG must have at least 2 nodes"
        assert not instantaneous_edges, "Only lagged graphs are supported for now"

        forecast_vars, covariates = split

        if instantaneous_edges:
            inst_G = self.generate_dag(num_nodes, intra_degree)
        else:
            inst_G = np.zeros((num_nodes, num_nodes))

        check_dagness(inst_G)

        lagged_G = np.zeros((lag, num_nodes, num_nodes))

        """
            expected_lag_edges = inter_degree * num_nodes
            total_lag_edges = num_nodes * num_nodes * lag

            edge_probs = expected_lag_edges / total_lag_edges
        """
        edge_probs = inter_degree / (num_nodes * lag)

        total_edges_per_lag_graph = num_nodes**2
        expected_edges_per_lag_graph = inter_degree * num_nodes
        edge_probs = float(expected_edges_per_lag_graph) / total_edges_per_lag_graph

        for l in range(lag):
            lagged_G[l, :, :] = (
                self.random.uniform(size=(num_nodes, num_nodes)) < edge_probs
            )

        # Make sure there is at least one historical parent
        zero_historical_parent_nodes = np.where(np.sum(lagged_G, axis=(0, 1)) == 0)[0]
        if len(zero_historical_parent_nodes) > 0:
            lag_idxs = self.random.choice(lag, len(zero_historical_parent_nodes))
            random_parent_idxs = self.random.choice(
                num_nodes, len(zero_historical_parent_nodes)
            )
            lagged_G[lag_idxs, random_parent_idxs, zero_historical_parent_nodes] = 1

        # Ensure each forecast variable has a historical parent in the covariate set for every lag step
        for l in range(lag):
            # Get all connections from covariate_{t-l} to forecast_{t}
            lag_submatrix = lagged_G[l][
                np.ix_(covariates, forecast_vars)
            ]  # (len(covariates), len(forecast_vars))

            # Check if there is at least one connection from covariate_{t-l} to forecast_{t}
            has_covariate_parent = lag_submatrix.sum(axis=0) > 0.0

            # For any forecast variable without a covariate parent, add a random covariate parent
            for i, node in enumerate(forecast_vars):
                if not has_covariate_parent[i]:
                    random_parent = self.random.choice(covariates)
                    lagged_G[l, random_parent, node] = 1

        complete_graph = np.concatenate((inst_G[None], lagged_G), axis=0)
        return complete_graph

    def init_weights(self, full_graph):
        """
        Initialize the weighted adjacency matrix for the linear model
        """
        # Sample weights from [-2., 0.5] U [0.5, 2.]
        weights = self.random.uniform(0.5, 1.5, size=full_graph.shape)
        weights[
            self.random.rand(*weights.shape) < 0.5
        ] *= -1  # Half of the weights are negative

        weighted_adjacency_matrix = full_graph * weights
        return weighted_adjacency_matrix


class BivariateCategoricalLinSVARBaseTask(CausalUnivariateCRPSTask):
    """
    A task where there are two variables X and Y
    X_t depends linearly on X_{t-1:t-L} and Y_{t-1:t-L} + gaussian noise
    Y are categorical variables and undergo distributional shifts at pred time

    E.g., Y =
    [00101020001010202... | 113435343534433534...]
    [hist.. | pred...]
    """

    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        """
        Currently the causal config is hardcoded. There are some heuristics for what are "good values" and what are not.

        Rule of thumb:

        lag:
            larger the lag, higher the increase in magnitude of the variable as we unroll in time. For large lag values we risk exploding the values of the time series.
        time_window:
            Larger the value, higher the chance of time series blowing up to NaNs (for which checks are in place).
        num_nodes:
            This is a bivariate task so it should always be 2. But in future if we increase this, we risk blowing up faster.
        noise_scale:
            Controls the noise levels of the model. Making it smaller makes the dynamics less stochastic, but allows us to accommodate larger lag and time_window.
        max_data_gen_trials:
            In case the data blows up due to any of these conditions, there are checks in place to catch the error and retry data generation with a different set of edge weights,
            and it would retry for a maximum of max_data_gen_trials.
        """

        self.causal_config = {
            "num_nodes": 2,
            "lag": 3,
            "instantaneous_edges": False,
            "burn_in": 30,
            "noise_type": "gauss",
            "noise_scale": 0.1,
            "time_window": 160,
            "max_data_gen_trials": 100,
            "num_forecast_vars": 1,  # CODE DOES NOT SUPPORT MULTIPLE FORECAST VARIABLES
        }
        assert (
            self.causal_config["num_forecast_vars"] == 1
        ), "Only 1 forecast variable supported"
        super().__init__(seed=seed, fixed_config=fixed_config)

    def generate_regimes(
        self, T, values, value_type, double_regimes, regime_limits=(10, 20)
    ):
        array = []
        regime_lengths = []
        regime_values = []
        remaining_length = T
        assert value_type in ["fluctuate", "const"]

        time_elapsed = 0
        ha = 0
        while remaining_length > 0:
            if value_type == "fluctuate":
                regime_length = self.random.randint(
                    low=regime_limits[0], high=regime_limits[1]
                )
                regime_length = min(
                    regime_length, remaining_length
                )  # Ensure it doesn't exceed remaining length
            else:
                regime_length = remaining_length

            regime_value = self.random.choice(values)
            ha += 1
            # Halfway point
            if double_regimes:
                if (
                    remaining_length > T // 2
                    and remaining_length - regime_length <= T // 2
                ):
                    first_half = T // 2 - time_elapsed
                    second_half = regime_length - first_half

                    regime_values.append(regime_value)
                    regime_values.append(regime_value * 2)

                    array.extend([regime_value] * first_half)
                    array.extend([regime_value * 2] * second_half)

                    regime_lengths.append(first_half)
                    regime_lengths.append(second_half)

                elif remaining_length <= T // 2:
                    regime_values.append(regime_value * 2)
                    regime_lengths.append(regime_length)
                    array.extend([regime_value * 2] * regime_length)

                else:
                    regime_values.append(regime_value)
                    regime_lengths.append(regime_length)
                    array.extend([regime_value] * regime_length)
            else:
                array.extend([regime_value] * regime_length)
                regime_values.append(regime_value)
                regime_lengths.append(regime_length)

            time_elapsed += regime_length
            remaining_length -= regime_length

        return np.array(array), (regime_values, regime_lengths)

    def generate_time_series(
        self, W, history_length, n_samples, noise_type="gauss", noise_scale=0.1
    ):
        d = self.causal_config["num_nodes"]
        L = self.causal_config["lag"]
        total_length = self.causal_config["burn_in"] + 2 * n_samples
        pred_length = n_samples - history_length

        hist_value_type = "fluctuate" if self.fluctuate_history else "const"
        future_val_type = "fluctuate"

        historical_covariates, hist_regime_details = self.generate_regimes(
            T=total_length - n_samples + history_length,
            values=[2, 8, 12, 20],
            value_type=hist_value_type,
            double_regimes=False,
            regime_limits=(40, 60),
        )

        future_covariates, pred_regime_details = self.generate_regimes(
            T=pred_length,
            values=[10, 12, 30, 40],
            value_type=future_val_type,
            double_regimes=True,
            regime_limits=(10, 20),
        )

        hist_regime_values, hist_regime_lengths = hist_regime_details
        trunc_hist_values, trunc_hist_lengths = truncate_regime(
            hist_regime_values, hist_regime_lengths, max_length=history_length
        )

        hist_start_timestamp = datetime.strptime(self.start_date, "%Y-%m-%d")
        hist_cov_desc = verbalize_variable_values(
            trunc_hist_values,
            trunc_hist_lengths,
            current_date=hist_start_timestamp,
            increment="daily",
        )

        num_hist_days = trunc_hist_lengths.sum().item()
        pred_start_timestamp = hist_start_timestamp + timedelta(days=num_hist_days)

        pred_regime_values, pred_regime_lengths = pred_regime_details
        pred_cov_desc = verbalize_variable_values(
            pred_regime_values,
            pred_regime_lengths,
            current_date=pred_start_timestamp,
            increment="daily",
        )

        covariate_values = np.concatenate(
            [historical_covariates, future_covariates]
        )  # (total_length, )

        inst_W, lagged_W = W[0], W[1:]
        lagged_W = lagged_W.reshape(L * d, d)
        assert inst_W.shape == (d, d)

        X = np.zeros([total_length, d])
        # idx 0 is covariate and idx 1 is causal variable
        X[:, 0] = covariate_values

        Xlags = np.zeros([total_length, L, d])
        Xlags = Xlags.reshape(total_length, L * d)

        g_intra = nx.DiGraph(inst_W)
        g_inter = nx.bipartite.from_biadjacency_matrix(
            csr_matrix(lagged_W), create_using=nx.DiGraph
        )

        ordered_vertices = list(nx.topological_sort(g_intra))

        for t in range(total_length):
            for j in ordered_vertices:
                if j in [0]:
                    pass

                else:
                    parents = list(g_intra.predecessors(j))
                    parents_prev = list(g_inter.predecessors(j + L * d))

                    X[t, j] = (
                        +X[t, parents] @ inst_W[parents, j]
                        + Xlags[t, parents_prev] @ lagged_W[parents_prev, j]
                    )

                    if noise_type == "gauss":
                        X[t, j] = X[t, j] + self.random.normal(scale=noise_scale)
                    elif noise_type == "exp":
                        X[t, j] = X[t, j] + self.random.exponential(scale=noise_scale)
                    elif noise_type == "gumbel":
                        X[t, j] = X[t, j] + self.random.gumbel(scale=noise_scale)

            if (t + 1) < total_length:
                Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * L]

        X_post_burn_in = X[-n_samples:, :]
        return X_post_burn_in, historical_covariates[0], (hist_cov_desc, pred_cov_desc)

    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data
        """
        d = self.causal_config["num_nodes"]
        L = self.causal_config["lag"]
        max_data_gen_trials = self.causal_config["max_data_gen_trials"]
        n_samples = self.causal_config["time_window"]
        noise_type = self.causal_config["noise_type"]
        noise_scale = self.causal_config["noise_scale"]

        if not hasattr(self, "start_date"):
            self.start_date = self.random.choice(
                pd.date_range("2021-01-01", "2028-01-01").strftime("%Y-%m-%d")
            )

        attempt = 0
        simulate_flag = True

        X_post_burn_in = None
        full_graph = None
        W = None

        vars_ = set(np.arange(d).tolist())

        # Only 1 variable to forecast, and it has to be the last variable in the df
        # for CRPS Evaluation: Hardcoded for now
        forecast_variable = d - 1
        covariate_idxs = vars_ - set([forecast_variable])
        assert len(covariate_idxs) == 1  # Bivariate setting has only 1 `covariate`

        full_graph = np.zeros((L + 1, d, d))
        full_graph[1:, :, forecast_variable] = 1

        history_length = int(0.8 * n_samples)
        pred_length = n_samples - history_length

        while simulate_flag:
            attempt += 1
            max_data_gen_trials -= 1
            if max_data_gen_trials < 0:
                simulate_flag = False
                raise ValueError(
                    f"Could not generate data after {max_data_gen_trials} trials"
                )

            try:
                W = self.init_weights(full_graph)

                X_post_burn_in, const_hist_value, cov_desc = self.generate_time_series(
                    W, history_length, n_samples, noise_type, noise_scale
                )

                # Check if X_post_burn_in has NaNs
                if np.isnan(X_post_burn_in).any():
                    continue

            except (OverflowError, FloatingPointError):
                continue

            # Find the max of X_post_burn_in
            if np.max(np.abs(X_post_burn_in)) < 1e4 or max_data_gen_trials <= 0:
                simulate_flag = False

        if max_data_gen_trials <= 0:
            raise ValueError(
                f"Could not simulate data, consider reducing noise, time window, or weight magnitudes"
            )

        # Split X_post_burn_in into history and future
        full_time_series_df = pd.DataFrame(X_post_burn_in)
        data_range = full_time_series_df[1].max() - full_time_series_df[1].min()
        full_time_series_df[
            1
        ] /= data_range  # normalize by data range for only the forecast variable.

        self.past_time = full_time_series_df[:history_length]
        self.future_time = full_time_series_df[history_length:]

        # Generate and set arbitrary timestamps
        ts = generate_timestamps(
            num_days=len(X_post_burn_in), start_date=self.start_date
        )
        self.past_time.index = pd.to_datetime(ts[:history_length])
        self.future_time.index = pd.to_datetime(ts[history_length:])

        self.graph = W
        self.historical_parents = get_historical_parents(full_graph)

        # Set scenario, constraints and background
        ""
        background = f"Given are variables X_0 and X_1, where X_0 is a covariate and X_1 is the variable to forecast."
        background += f" Variables are generated from a linear Structural Vector Autoregressive (SVAR) model with additive {noise_type} noise and a noise scale of {noise_scale / data_range:.3e}, with lag = {L}."
        self.background = background

        self.scenario = self.get_scenario(
            const_hist_value, history_length, pred_length, cov_desc
        )
        self.scenario += " " + self.get_causal_context(W, L)
        self.constraints = None

    def get_scenario(self, const_hist_value, history_length, pred_length, cov_desc):
        hist_cov_desc_list, pred_cov_desc_list = cov_desc
        pred_cov_desc = ", ".join(pred_cov_desc_list)

        line1 = f"The task is to forecast the value of the variable X_{self.causal_config['num_nodes'] - 1} at time t, given the values of the covariate X_0 and the variable X_{self.causal_config['num_nodes'] - 1} itself at times t-1, ... t-{self.causal_config['lag']}."

        if self.fluctuate_history:
            hist_val = ", ".join(hist_cov_desc_list)
            line2 = f"For the first {history_length} days, the covariate X_0 takes a value of {hist_val}."
        else:
            line2 = f"For the first {history_length} days, the covariate X_0 is constant at {const_hist_value}."

        line3 = f"For the next {pred_length} days, the covariate X_0 takes a value of {pred_cov_desc}. Each day can be treated as a timestep for the forecasting task."

        scenario = f"{line1}\n{line2}\n{line3}"
        return scenario

    @abstractmethod
    def get_causal_context(self, W, lag):
        pass


"""
    Causal Context Level 1
    ----------------------

    Parents of the forecast variable are given. Edge weights of the linear SVAR model are not given and have to be inferred,
    by solving the linear SVAR model which can be done by observing the parent history and child history. The inferred edge weights
    should then used to forecast the future values of the forecast variable, given the future values of the covariate.

    Causal context example:
    Parents for X_1 at lag 1: X_0, X_1
    Parents for X_1 at lag 2: X_0, X_1
"""


class MinimalCausalContextBivarLinSVAR(BivariateCategoricalLinSVARBaseTask):

    _context_sources = ["c_cov", "c_causal"]
    _skills = BivariateCategoricalLinSVARBaseTask._skills + [
        "reasoning: math",
        "reasoning: causal",
        "retrieval: memory",
    ]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        self.fluctuate_history = True
        self.plot_name = "MinimalContextBivarCatLinSVAR"
        super().__init__(fixed_config, seed)

    def get_causal_context(self, W, lag):
        d = W.shape[-1]
        graph_desc = []
        for i in range(d):
            if i == d - 1:
                desc_i = parent_descriptions(W, lag, i, desc="minimal")
            else:
                desc_i = f"No parents for X_{i} at any lag."
            graph_desc.append(desc_i)

        textual_causal_desc = "\n".join(graph_desc)
        causal_context = (
            f"The causal parents affect the child variables at different lags.\n"
        )
        causal_context += f"The complete set of causal parents for each variable is given below, and there are no confounders.\n{textual_causal_desc}\n"
        return causal_context


"""
    Causal Context Level 2
    ----------------------

    Parents of the forecast variable are given along with the exact edge weights.

    Causal context example:
    Parents for X_1 at lag 1: X_0 affect the forecast variable as 0.5 * X_0 + 0.5 * X_1.
    Parents for X_1 at lag 2: X_0 affect the forecast variable as 0.73 * X_0 + 0.52 * X_1.

    In the expression, the timesteps of X_0 and X_1 have to be inferred from the text description (which mentions `at lag <l>`).
    Thus, the model must implictly reason that X_0^{t} = 0.5 * X_0^{t-1} + 0.5 * X_1^{t-1} + 0.73 * X_0^{t-2} + 0.52 * X_1^{t-2} + \epsilon
    \epsilon has to be inferred from details on noise_scale, distribution and the linear SVAR model mentioned in task.background.
"""


class FullCausalContextImplicitEquationBivarLinSVAR(
    BivariateCategoricalLinSVARBaseTask
):
    _context_sources = ["c_cov", "c_causal"]
    _skills = BivariateCategoricalLinSVARBaseTask._skills + [
        "reasoning: math",
        "reasoning: causal",
        "retrieval: memory",
    ]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        self.fluctuate_history = True
        self.plot_name = "FullContextImplicitBivarCatLinSVAR"
        super().__init__(fixed_config, seed)

    def get_causal_context(self, W, lag):
        d = W.shape[-1]
        graph_desc = []
        for i in range(d):
            if i == d - 1:
                desc_i = parent_descriptions(
                    W, lag, i, desc="edge_weights_implicit_equation"
                )
            else:
                desc_i = f"No parents for X_{i} at any lag."
            graph_desc.append(desc_i)

        textual_causal_desc = "\n".join(graph_desc)
        causal_context = (
            f"The causal parents affect the child variables at different lags.\n"
        )
        causal_context += f"The causal parents for each variable is given below:\n{textual_causal_desc}"
        return causal_context


"""
    Causal Context Level 3
    ----------------------

    Parents of the forecast variable are given along with the exact edge weights.

    Causal context example:
    The causal expression for X_1 is completely described by the following equation:
        X_1^{t} = 0.5 * X_0^{t-1} + 0.5 * X_1^{t-1} + 0.73 * X_0^{t-2} + 0.52 * X_1^{t-2} + \epsilon

    In the expression, the mathematical relations are given explicitly and the task boils down to mathematical reasoning.
    The exact value/distribution of \epsilon still has to be inferred from textual descriptions of noise_scale and distribution mentioned in task.background.
"""


class FullCausalContextExplicitEquationBivarLinSVAR(
    BivariateCategoricalLinSVARBaseTask
):
    _context_sources = ["c_cov", "c_causal"]
    _skills = BivariateCategoricalLinSVARBaseTask._skills + [
        "reasoning: math",
        "reasoning: causal",
    ]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(self, fixed_config: dict = None, seed: int = None):
        self.fluctuate_history = True
        self.plot_name = "FullContextExplicitBivarCatLinSVAR"
        super().__init__(fixed_config, seed)

    def get_causal_context(self, W, lag):
        d = W.shape[-1]
        graph_desc = []
        for i in range(d):
            desc_i = parent_descriptions(
                W, lag, i, desc="edge_weights_explicit_equation"
            )

            graph_desc.append(desc_i)

        graph_desc.append(
            "\epsilon_{i}^{t} given in the equations corresponds to the noise variable with the given noise scale, for X_i^{t}"
        )
        textual_causal_desc = "\n".join(graph_desc)
        causal_context = (
            f"The causal parents affect the child variables at different lags.\n"
        )
        causal_context += f"The mathematical equations describing the cause-effect relationship for each variable is given below:\n{textual_causal_desc}."
        return causal_context


__TASKS__ = [
    MinimalCausalContextBivarLinSVAR,  # Level 1: Parents are given
    FullCausalContextImplicitEquationBivarLinSVAR,  # Level 2: Parents and edge weights are given, details on timesteps in the SVAR expression and epsilon have to be inferred from text
    FullCausalContextExplicitEquationBivarLinSVAR,  # Level 3: Parents and edge weights are given along with explicit timestep indexing in the SVAR equation and epsilon have to be inferred from text
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            MinimalCausalContextBivarLinSVAR,
            FullCausalContextImplicitEquationBivarLinSVAR,
            FullCausalContextExplicitEquationBivarLinSVAR,
        ],
    )
]
