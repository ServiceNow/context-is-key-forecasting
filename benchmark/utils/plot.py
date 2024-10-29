import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_task(task):
    """
    Plots a task's numerical data and returns the figure.

    Parameters:
    -----------
    task: BaseTask
        The task to plot

    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure containing the plot

    """
    plt.figure()

    # Prepare data
    past_timesteps = task.past_time.index
    past_values = task.past_time.to_numpy()[:, -1]
    future_timesteps = task.future_time.index
    future_values = task.future_time.to_numpy()[:, -1]

    # The fill_between method is only ok with pd.DatetimeIndex
    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    timesteps = np.concatenate([past_timesteps, future_timesteps])
    values = np.concatenate([past_values, future_values])

    # Plot the history
    plt.plot(
        past_timesteps,
        past_values,
        color=(0, 0, 0),
        linewidth=2,
        zorder=5,
        label="History",
    )

    # Plot the forecast
    plt.plot(
        future_timesteps,
        future_values,
        color="orange",
        linewidth=2,
        zorder=5,
        label="Ground Truth",
    )

    # Shade the entire future region in light green to indicate the forecast region
    y_min, y_max = plt.ylim()
    # Shade forecast window
    plt.fill_between(
        future_timesteps,
        -99999,  # Use the current minimum y-limit
        99999,  # Use the current maximum y-limit
        facecolor=(0.9, 1, 0.9),
        interpolate=True,
        label="Forecast",
        zorder=0,
    )
    # Shade RoI
    if type(task.region_of_interest) != type(None):
        if isinstance(task.region_of_interest, list):
            # Convert the list of timesteps to a Pandas Series and find contiguous groups
            roi_series = pd.Series(task.region_of_interest)
            contiguous_regions = []
            start_idx = roi_series.iloc[0]
            for i in range(1, len(roi_series)):
                if roi_series.iloc[i] != roi_series.iloc[i - 1] + 1:
                    contiguous_regions.append(slice(start_idx, roi_series.iloc[i - 1]))
                    start_idx = roi_series.iloc[i]
            contiguous_regions.append(slice(start_idx, roi_series.iloc[-1]))
        elif isinstance(task.region_of_interest, int):
            contiguous_regions = [[task.region_of_interest]]
        else:
            contiguous_regions = [task.region_of_interest]
        for region_index, region in enumerate(contiguous_regions):
            plt.fill_between(
                future_timesteps[region],
                -99999,  # Use the current minimum y-limit
                99999,  # Use the current maximum y-limit
                facecolor=(0, 0.8, 0),
                interpolate=True,
                label="Region of Interest" if region_index == 0 else None,
                zorder=0,
            )

    # Set the plot limits to ensure no white space around the shading
    plt.ylim(y_min, y_max)
    plt.xlim(timesteps[0], timesteps[-1])

    # Minor style tweaks
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()
