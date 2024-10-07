import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import traceback
from pprint import pprint

from collections import defaultdict
from functools import partial
from pathlib import Path

from . import ALL_TASKS
from .config import DEFAULT_N_SAMPLES
from .utils.cache import ResultCache, CacheMissError


logger = logging.getLogger("Evaluation")


def plot_forecast_univariate(task, samples, path, return_fig=False):
    """
    Plot the first variable of a forecast.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    samples: np.array
        The forecast of shape [samples, time dimension, number of variables]
    path: Pathlike
        Directory in which to save the figure

    """
    samples = samples[:, :, -1]

    future_timesteps = task.future_time.index
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    fig = task.plot()
    ax = fig.gca()

    for zorder, quant, color, label in [
        [1, 0.05, (0.75, 0.75, 1), "5%-95%"],
        [2, 0.10, (0.25, 0.25, 1), "10%-90%"],
        [3, 0.25, (0, 0, 0.75), "25%-75%"],
    ]:
        lower_quantile = np.quantile(samples, quant, axis=0).astype(float)
        upper_quantile = np.quantile(samples, 1 - quant, axis=0).astype(float)

        ax.fill_between(
            future_timesteps,
            lower_quantile,
            upper_quantile,
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )

        if quant == 0.05:
            min_quantile_value = np.min(lower_quantile)
            max_quantile_value = np.max(upper_quantile)

    ax.plot(
        future_timesteps,
        np.quantile(samples, 0.5, axis=0),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    target_name = task.past_time.columns[-1]
    ax.set_ylim(
        np.min(
            [
                np.min(task.past_time[target_name]),
                np.min(task.future_time[target_name]),
                min_quantile_value,
            ]
        ),
        np.max(
            [
                np.max(task.past_time[target_name]),
                np.max(task.future_time[target_name]),
                max_quantile_value,
            ]
        ),
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1, 3, 4, 5, 6]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax.set_title(task.name)

    fig.savefig(path / "forecast.pdf")
    fig.savefig(path / "forecast.png", bbox_inches="tight")
    if return_fig:
        return fig
    else:
        plt.close(fig)


def save_context(task, path):
    """
    Save the context of a task to a file for future reference.

    """
    with open(path / "context", "w") as f:
        f.write(
            f"""
Background:
{task.background}

Constraints:
{task.constraints}

Scenario:
{task.scenario}
"""
        )


def save_evaluation(evaluation, path):
    """
    Save the content of the task evaluation content to a file for future reference.
    """
    with open(path / "evaluation", "w") as f:
        pprint(evaluation, f)


def save_extra_info(extra_info, path):
    """
    Save the content of the model extra information content to a file for future reference.
    """
    with open(path / "extra_info", "w") as f:
        pprint(extra_info, f)


def evaluate_task(
    task_cls,
    seed,
    method_callable,
    n_samples,
    output_folder=None,
):
    if output_folder:
        task_folder = output_folder / task_cls.__name__
        seed_folder = task_folder / f"{seed}"
        seed_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Instantiate the task
        task = task_cls(seed=seed)

        logger.info(f"Method {method_callable} - Task {task.name} - Seed {seed}")
        samples = method_callable(task_instance=task, n_samples=n_samples)
        if isinstance(samples, tuple):
            samples, extra_info = samples
        else:
            extra_info = {}
        evaluation = task.evaluate(samples)
        result = {
            "seed": seed,
            "score": (
                evaluation["metric"] if isinstance(evaluation, dict) else evaluation
            ),
        }

        if output_folder:
            # Save forecast plots
            plot_forecast_univariate(task=task, samples=samples, path=seed_folder)

            # Save context
            save_context(task=task, path=seed_folder)

            # Save metric content
            save_evaluation(evaluation=evaluation, path=seed_folder)

            # Save extra_info content
            save_extra_info(extra_info=extra_info, path=seed_folder)

        return (task_cls.__name__, result)

    except CacheMissError:
        logger.info(f"Skipping over cache miss.")
        return (
            task_cls.__name__,
            {
                "seed": seed,
                "error": f"Cache miss - Method {method_callable} - Task {task_cls.__name__} - Seed {seed}",
            },
        )

    except Exception as e:
        logger.error(f"Error evaluating task {task_cls.__name__} - Seed {seed}: {e}")
        logger.error(traceback.format_exc())
        if output_folder:
            with open(seed_folder / "error", "w") as f:
                f.write(str(e))
                f.write("\n")
                f.write(traceback.format_exc())
        return (task_cls.__name__, {"seed": seed, "error": str(e)})


def evaluate_all_tasks(
    method_callable,
    seeds=5,
    n_samples=DEFAULT_N_SAMPLES,
    output_folder=None,
    use_cache=True,
    cache_name=None,
    max_parallel=None,
    skip_cache_miss=False,
):
    """
    Evaluate a method on all tasks.

    Parameters:
    -----------
    method_callable: callable
        The method to evaluate. Must take a task instance and return samples.
    seeds: int
        Number of seeds to evaluate on each task.
    n_samples: int
        Number of samples to generate.
    output_folder: Pathlike
        Directory in which to save the results.
    use_cache: bool
        Whether to use a cache to store/load results.
    cache_name: str
        Name of the cache.
    max_parallel: int
        Number of parallel processes to use.
    skip_cache_miss: bool
        Whether to skip computing tasks that are not found in the cache (useful for report generation).

    """

    logger.info(
        f"Evaluating method {method_callable} with {seeds} seeds and {n_samples} samples on {len(ALL_TASKS)} tasks."
    )

    if output_folder:
        logger.info(f"Saving outputs to {output_folder}")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("No output folder provided. Results will not be saved.")

    if use_cache:
        method_callable = ResultCache(
            method_callable, method_name=cache_name, raise_on_miss=skip_cache_miss
        )

    tasks_to_evaluate = []
    for task_cls in ALL_TASKS:
        for seed in range(1, seeds + 1):
            tasks_to_evaluate.append((task_cls, seed))

    func = partial(
        evaluate_task,
        method_callable=method_callable,
        n_samples=n_samples,
        output_folder=output_folder,
    )

    if max_parallel == 1:
        # No parallelism, just evaluate tasks in a loop
        results_list = [func(task_cls, seed) for task_cls, seed in tasks_to_evaluate]
    else:
        # Use multiprocessing to parallelize the evaluation
        with multiprocessing.Pool(processes=max_parallel) as pool:
            results_list = pool.starmap(func, tasks_to_evaluate)

    # Collect results
    results = defaultdict(list)
    for task_name, result in results_list:
        results[task_name].append(result)

    return results
