import networkx as nx
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import scipy
from datetime import datetime, timedelta


def check_dagness(inst_graph):
    """
    Check for acylcity of the instantaneous graph
    """
    num_nodes = inst_graph.shape[0]
    error_message = colored(
        "Error in DAG generation: Instantaneous graph is not acyclic!", "red"
    )
    dag_constraint = np.trace(scipy.linalg.expm(inst_graph * inst_graph)) - num_nodes
    assert dag_constraint == 0, error_message


def parent_descriptions(W, L, node, desc):
    """
    Given the DAGs, returns a textual description of the parents
    of a single node.
    """
    res = ""
    coeff_parent_vars = []

    for l in range(1, L + 1):
        parents = W[l, :, node]
        # Get index
        parents = np.where(parents != 0)[0]
        if len(parents) == 0:
            if desc in ["minimal", "edge_weights_implicit_equation"]:
                res += f"No parents for X_{node} at lag {l}.\n"

        elif desc == "minimal":
            parent_vars = []
            for parent in parents:
                parent_vars.append(f"X_{parent}")

            parent_str = ", ".join(parent_vars)
            res += f"Parents for variable X_{node} at lag {l}: {parent_str}.\n"

        elif desc == "edge_weights_implicit_equation":
            parent_vars = []
            coeff_parent_vars = []
            for parent in parents:
                coefficient = W[l, parent, node]
                coeff_parent_vars.append(f"{coefficient:.3f} * X_{parent}")
                parent_vars.append(f"X_{parent}")

            expression = " + ".join(coeff_parent_vars)
            res += f"Parents for X_{node} at lag {l}: {parent_vars} affect the forecast variable as {expression}.\n"

        elif desc == "edge_weights_explicit_equation":
            for parent in parents:
                coefficient = W[l, parent, node]
                timestep_indexed_parent = f"X_{parent}" + "^{t-" + str(l) + "}"
                coeff_parent_vars.append(
                    f"{coefficient:.3f} * {timestep_indexed_parent}"
                )

        else:
            NotImplementedError(
                "`desc` should be `minimal`, `edge_weights_implicit_equation`, or `edge_weights_plicit_equation`"
            )

    if desc == "edge_weights_explicit_equation":
        if len(coeff_parent_vars) == 0:
            res = f"X_{node}^" + "{t} = \epsilon_" + f"{node}^" + "{t}"
        else:
            expression = " + ".join(coeff_parent_vars)
            res = f"X_{node}^" + "{t} = " + expression + f" + \epsilon_{node}^" + "{t}"

    return res


def get_historical_parents(full_graph):
    num_nodes = full_graph.shape[-1]
    # For each variable (column), tells which nodes are parents aggregated across the lag timesteps
    historical_parent_matrix = np.sum(full_graph[1:], axis=0) > 0

    historical_parents = []
    for i in range(num_nodes):
        parents = np.where(historical_parent_matrix[:, i])[0]
        historical_parents.append(parents.tolist())

    assert len(historical_parents) == num_nodes
    return historical_parents


def generate_timestamps(num_days, start_date="2025-06-01"):
    timestamps = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")

    for _ in range(num_days):
        timestamps.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))
        current_date += timedelta(days=1)

    return timestamps


def verbalize_variable_values(regime_values, regime_lengths, current_date, increment):
    assert increment == "daily"
    pred_time_covariate_desc = []

    for i in range(len(regime_values)):
        from_date_str = current_date.strftime("%Y-%m-%d")
        current_date += timedelta(days=int(regime_lengths[i]) - 1)
        to_date_str = current_date.strftime("%Y-%m-%d")
        text = f"{regime_values[i]} from {from_date_str} to {to_date_str}"
        pred_time_covariate_desc.append(text)
        current_date += timedelta(days=1)

    return pred_time_covariate_desc


def truncate_regime(regime_values, regime_lengths, max_length=100):
    """
    Truncate a list regime_values and regime_lengths given maximum cumulative regime length
    """
    regime_lengths = np.array(regime_lengths)
    cum_rev_regime_lengths = np.cumsum(regime_lengths[::-1])
    border_idx = np.argwhere(cum_rev_regime_lengths > max_length)[0, 0]
    num_elements_to_keep = border_idx + 1

    trunc_regime_values = regime_values[-num_elements_to_keep:].copy()
    trunc_regime_lengths = regime_lengths[-num_elements_to_keep:].copy()
    trunc_regime_lengths[0] = max_length - trunc_regime_lengths[1:].sum()
    assert trunc_regime_lengths.sum() == max_length
    return trunc_regime_values, trunc_regime_lengths


def plot_temporal_graph(complete_graph):
    """
    Function to visualize the instantaneous and lagged graphs, to aid in debugging
    """
    L = complete_graph.shape[0] - 1
    d = complete_graph.shape[1]
    G = nx.DiGraph()

    for lag in range(L + 1):
        for i in range(d):
            G.add_node((lag, i), label=f"{i}")

    # Add edges based on the adjacency matrices
    for lag in range(L + 1):
        for i in range(d):
            for j in range(d):
                if complete_graph[lag, i, j] == 1:
                    G.add_edge(
                        (L - lag, i), (L, j)
                    )  # Connection from (t-lag, variable i) to (t, variable j)

    pos = {}
    for lag in range(L + 1):
        for i in range(d):
            pos[(lag, i)] = (lag, -i)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={(lag, i): f"{i}" for lag in range(L + 1) for i in range(d)},
        node_size=500,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        ax=ax,
    )

    # Set labels for each column
    timestep_labels = [f"T-{L-lag}" if lag != L else "T" for lag in range(L + 1)]
    for lag, label in enumerate(timestep_labels):
        ax.text(
            lag,
            -d + 0.3,
            label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight="bold",
            color="green",
        )

    ax.set_title("Temporal Adjacency Matrix Visualization")
    plt.show()
    plt.savefig("temporal_graph.png")
