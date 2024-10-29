import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

from benchmark.sensor_maintenance import SensorMaintenanceInPredictionTask


def plot_forecast_univariate(task, filename):
    """
    Plot the first variable of a forecast.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    filename: Pathlike
        Where to save the figure
    """
    past_timesteps = task.past_time.index
    past_values = task.past_time.to_numpy()[:, -1]
    future_timesteps = task.future_time.index
    future_values = task.future_time.to_numpy()[:, -1]

    # The fill_between method is only ok with pd.DatetimeIndex
    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    plt.figure(figsize=(15, 15))

    plt.plot(
        past_timesteps,
        past_values,
        color=(1, 0, 0),
        linewidth=2,
        zorder=5,
        label="history",
    )
    plt.plot(
        future_timesteps,
        future_values,
        color=(0, 0, 1),
        linewidth=2,
        zorder=5,
        label="prediction",
    )
    plt.xticks(fontsize=7)

    plt.legend()

    plt.title(
        "\n".join(textwrap.wrap(task.scenario, width=40))
    )  # Change this to task.background or task.constraint depending on task
    plt.savefig(filename)


def plot_forecast_with_covariates(task, filename):
    """
    Plot the first variable of a forecast, along with all covariates.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    filename: Pathlike
        Where to save the figure
    """
    past_values = task.past_time.to_numpy()[:, -1]
    future_values = task.future_time.to_numpy()[:, -1]

    past_timesteps = np.arange(len(past_values))
    future_timesteps = np.arange(
        len(past_values), len(past_values) + len(future_values)
    )

    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    past_covariates = task.past_time.to_numpy()[:, :-1]
    future_covariates = task.future_time.to_numpy()[:, :-1]
    num_covariates = past_covariates.shape[1]

    fig, axes = plt.subplots(
        num_covariates + 1, 1, figsize=(18, 12 * num_covariates), sharex=True
    )
    plt.xticks(fontsize=7)

    for k in range(num_covariates + 1):

        if k == 0:  # Forecast variable
            past, future = past_values, future_values
            title = "Forecast variable"
        else:
            past, future = past_covariates[:, k - 1], future_covariates[:, k - 1]
            title = f"Covariate {k}"

        past_line = axes[k].plot(
            past_timesteps,
            past,
            color=(1, 0, 0),
            linewidth=2,
            zorder=5,
            label="history",
        )

        future_line = axes[k].plot(
            future_timesteps,
            future,
            color=(0, 0, 1),
            linewidth=2,
            zorder=5,
            label="prediction",
        )

        axes[k].grid()
        axes[k].set_title(title)

        if k == 0:
            lines = [past_line[0], future_line[0]]
            lables = ["history", "prediction"]

    # Add x-label to the last subplot
    axes[-1].set_xlabel("Time")

    fig.legend(lines, lables, loc="upper right")

    scenario = "\n".join(textwrap.wrap(task.scenario, width=125))
    background = "\n".join(textwrap.wrap(task.background, width=125))
    causal_context = "\n".join(textwrap.wrap(task.causal_context, width=125))

    # Count number of characters in scenario, background and causal context
    background_chars = len(background)
    scenario_chars = len(scenario)
    causal_context_chars = len(causal_context)

    plt.suptitle(
        f"Background ({background_chars}): {background}\nCausal context ({causal_context_chars}): {causal_context}\nScenario ({scenario_chars}): {scenario}",
        fontsize=16,
    )  # Change this to task.background or task.constraint depending on task
    plt.tight_layout()
    plt.savefig(filename)


task = SensorMaintenanceInPredictionTask()  # Change this to required task
plot_forecast_univariate(task, f"task_example_plotting.png")
