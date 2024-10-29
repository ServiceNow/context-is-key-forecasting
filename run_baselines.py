"""
Run all baselines on all tasks and save the results to a Pandas dataframe.

"""

import argparse
import json
import inspect
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path
from benchmark.baselines.direct_prompt import DirectPrompt
from benchmark.baselines.lag_llama import lag_llama
from benchmark.baselines.chronos import ChronosForecaster
from benchmark.baselines.moirai import MoiraiForecaster
from benchmark.baselines.llm_processes import LLMPForecaster
from benchmark.baselines.timellm import TimeLLMForecaster
from benchmark.baselines.unitime import UniTimeForecaster
from benchmark.baselines.timegen import timegen1
from benchmark.baselines.naive import oracle_baseline, random_baseline
from benchmark.baselines.statsmodels import (
    ExponentialSmoothingForecaster,
)
from benchmark.baselines.r_forecast import R_ETS, R_Arima
from benchmark.evaluation import evaluate_all_tasks
from benchmark.config import RESULT_CACHE_PATH


logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def experiment_naive(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Naive baselines (random and oracle)

    """
    results = []
    results.append(
        (
            "random",
            evaluate_all_tasks(
                random_baseline,
                n_samples=n_samples,
                output_folder=f"{output_folder}/random/",
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            ),
        )
    )
    results.append(
        (
            "oracle",
            evaluate_all_tasks(
                oracle_baseline,
                n_samples=n_samples,
                output_folder=f"{output_folder}/oracle/",
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            ),
        )
    )
    return results, {}


def experiment_lag_llama(
    n_samples, output_folder, max_parallel=10, skip_cache_miss=False
):
    """
    Lag LLAMA baseline

    """
    results = evaluate_all_tasks(
        lag_llama,
        n_samples=n_samples,
        output_folder=f"{output_folder}/lag_llama/",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    return results, {}


def experiment_chronos(
    model_size, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
):
    """
    Chronos baselines

    """
    results = evaluate_all_tasks(
        ChronosForecaster(model_size=model_size),
        n_samples=n_samples,
        output_folder=f"{output_folder}/chronos/",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    return results, {}


def experiment_moirai(
    model_size, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
):
    """
    Moirai baselines

    """
    results = evaluate_all_tasks(
        MoiraiForecaster(model_size=model_size),
        n_samples=n_samples,
        output_folder=f"{output_folder}/moirai/",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    return results, {}


def experiment_statsmodels(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Statsmodels baselines (Exponential Smoothing)

    """
    return (
        evaluate_all_tasks(
            ExponentialSmoothingForecaster(),
            n_samples=n_samples,
            output_folder=f"{output_folder}/exp_smoothing/",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def experiment_r_ets(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Baseline using the R "forecast" package: ETS

    """
    return (
        evaluate_all_tasks(
            R_ETS(),
            n_samples=n_samples,
            output_folder=f"{output_folder}/r_ets/",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def experiment_r_arima(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Baseline using the R "forecast" package: Arima

    """
    return (
        evaluate_all_tasks(
            R_Arima(),
            n_samples=n_samples,
            output_folder=f"{output_folder}/r_arima/",
            max_parallel=1,  # Hardcoded as it's buggy with None
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def experiment_directprompt(
    llm,
    use_context,
    n_samples,
    output_folder,
    max_parallel=1,
    skip_cache_miss=False,
    batch_size=None,
    batch_size_on_retry=5,
    n_retries=3,
    temperature=1.0,
):
    """
    DirectPrompt baselines

    """
    # Costs per 1000 tokens
    openai_costs = {
        "gpt-4o": {"input": 0.005, "output": 0.015},  # Same price Azure and OpenAI
        "gpt-35-turbo": {"input": 0.002, "output": 0.002},
        "gpt-3.5-turbo": {"input": 0.003, "output": 0.006},  # OpenAI API
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # OpenAI API
        "llama-3.1-405b": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3.1-405b-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-2-7B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-2-70B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-8B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-8B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-70B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-70B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "mixtral-8x7B": {"input": 0.0, "output": 0.0},  # Toolkit
        "mixtral-8x7B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "phi-3-mini-128k-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-9B": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-9B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-27B": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-27B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
    }
    if not llm.startswith("openrouter-") and llm not in openai_costs:
        raise ValueError(f"Invalid model: {llm} -- Not in cost dictionary")

    dp_forecaster = DirectPrompt(
        model=llm,
        use_context=use_context,
        token_cost=openai_costs[llm] if not llm.startswith("openrouter-") else None,
        batch_size=batch_size,
        batch_size_on_retry=batch_size_on_retry,
        n_retries=n_retries,
        temperature=temperature,
        dry_run=skip_cache_miss,
    )
    results = evaluate_all_tasks(
        dp_forecaster,
        n_samples=n_samples,
        output_folder=f"{output_folder}/{dp_forecaster.cache_name}",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    total_cost = dp_forecaster.total_cost
    del dp_forecaster

    return results, {"total_cost": total_cost}


def experiment_timellm(
    use_context,
    dataset,
    pred_len,
    n_samples,
    output_folder,
    max_parallel=1,
    skip_cache_miss=False,
):
    """
    TimeLLM baselines
    Doesn't use n_samples as it is not implemented in the TimeLLMForecaster

    """
    timellm_forecaster = TimeLLMForecaster(
        use_context=use_context,
        dataset=dataset,
        pred_len=pred_len,
        dry_run=skip_cache_miss,
    )

    return (
        evaluate_all_tasks(
            timellm_forecaster,
            n_samples=n_samples,
            output_folder=f"{output_folder}/{timellm_forecaster.cache_name}",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def experiment_unitime(
    use_context,
    pred_len,
    n_samples,
    output_folder,
    dataset="",
    per_dataset_checkpoint=False,
    max_parallel=1,
    skip_cache_miss=False,
):
    """
    TimeLLM baselines
    Doesn't use n_samples as it is not implemented in the TimeLLMForecaster

    """
    unitime_forecaster = UniTimeForecaster(
        use_context=use_context,
        dataset=dataset,
        pred_len=pred_len,
        per_dataset_checkpoint=per_dataset_checkpoint,
        dry_run=skip_cache_miss,
    )

    return (
        evaluate_all_tasks(
            unitime_forecaster,
            n_samples=n_samples,
            output_folder=f"{output_folder}/{unitime_forecaster.cache_name}",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def experiment_timegen1(
    n_samples, output_folder, max_parallel=10, skip_cache_miss=False
):
    """
    Nixtla TimeGEN-1 baseline

    """
    results = evaluate_all_tasks(
        timegen1,
        n_samples=n_samples,
        output_folder=f"{output_folder}/timegen1/",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    return results, {}


def experiment_llmp(
    llm, use_context, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
):
    """
    LLM Process baselines

    """
    llmp_forecaster = LLMPForecaster(
        llm_type=llm, use_context=use_context, dry_run=skip_cache_miss
    )
    return (
        evaluate_all_tasks(
            llmp_forecaster,
            n_samples=n_samples,
            output_folder=f"{output_folder}/{llmp_forecaster.cache_name}",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def compile_results(results, cap=None):
    # Compile results into Pandas dataframe
    errors = defaultdict(list)
    missing = defaultdict(list)
    results_ = {
        "Task": [task for task in list(results.values())[0]],
    }
    for method, method_results in results.items():
        _method_results = []
        for task in results_["Task"]:
            task_results = []

            for seed_res in method_results[task]:
                # Keep track of exceptions and missing results
                seed_res["task"] = task
                if "error" in seed_res:
                    if "cache miss" in seed_res["error"].lower():
                        missing[method].append(seed_res)
                    else:
                        errors[method].append(seed_res)
                else:
                    if cap == None:
                        score = seed_res["score"]
                    else:
                        score = min(seed_res["score"], cap)
                    task_results.append(score)

            mean = np.mean(task_results)
            std = np.std(task_results, ddof=1)
            stderr = std / np.sqrt(len(task_results))
            _method_results.append(f"{mean.round(3): .3f} ± {stderr.round(3) :.3f}")

        results_[method] = _method_results

    results = pd.DataFrame(results_).sort_values("Task").set_index("Task")
    del results_

    return results, missing, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=25,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results/",
        help="Output folder for results",
    )
    parser.add_argument(
        "--exp-spec",
        type=str,
        help="Experiment specification file",
    )
    parser.add_argument(
        "--list-exps",
        action="store_true",
        help="List available experiments and their parameters",
    )
    parser.add_argument(
        "--skip-cache-miss",
        action="store_true",
        help="Skip tasks that have not already been computed",
    )
    parser.add_argument(
        "--cap",
        type=float,
        help="Cap value to cap each instance's metric",
    )

    args = parser.parse_args()
    output_folder = Path(args.output)

    # List all available experiments
    if args.list_exps:
        print("Available experiments:")
        # Filter globals to only include functions that start with "experiment_"
        exp_funcs = [
            v
            for k, v in globals().items()
            if k.startswith("experiment_") and inspect.isfunction(v)
        ]

        # Print each experiment function name with a list of its parameters
        for func in exp_funcs:
            # Get the function signature
            signature = inspect.signature(func)
            # List of parameters excluding 'n_samples', 'output_folder', and 'skip_cache_miss'
            params = [
                name
                for name, param in signature.parameters.items()
                if name not in ["n_samples", "output_folder", "skip_cache_miss"]
            ]
            # Print the function name and its parameters
            print(f"\t{func.__name__}({', '.join(params)})")

        exit()

    # Run all experiments
    all_results = {}
    extra_infos = {}
    # ... load specifications
    with open(args.exp_spec, "r") as f:
        exp_spec = json.load(f)
    # ... run each experiment
    for exp in exp_spec:
        current_results = {}
        print(f"Running experiment: {exp['label']}")
        exp_label = exp["label"]
        # ... extract configuration
        config = {k: v for k, v in exp.items() if k != "method" and k != "label"}
        config["n_samples"] = args.n_samples
        config["output_folder"] = output_folder / exp_label
        config["skip_cache_miss"] = args.skip_cache_miss
        print(f"\tConfig: {config}")
        # ... do it!
        function = globals().get(f"experiment_{exp['method']}")
        # ... process results
        res, extra_info = function(**config)
        if isinstance(res, list):
            all_results.update({f"{exp_label}_{k}": v for k, v in res})
            current_results.update({f"{exp_label}_{k}": v for k, v in res})
        else:
            all_results[exp_label] = res
            current_results[exp_label] = res
        extra_infos[exp_label] = extra_info

        # Compile results
        current_results, missing, errors = compile_results(
            current_results, cap=args.cap
        )
        print(current_results)
        print("Number of missing results:", {k: len(v) for k, v in missing.items()})
        print("Number of errors:", {k: len(v) for k, v in errors.items()})

        # Save results to CSV. Note: Saved in output_folder / exp_label, not output_folder as it is exp-specific result
        filename = "results.csv" if not args.cap else f"results-cap-{args.cap}.csv"
        print(f"Saving results to {output_folder/exp_label}/{filename}")
        current_results.to_csv(output_folder / exp_label / filename)
        print(f"Saving missing results to {output_folder/exp_label}/missing.json")
        with open(output_folder / exp_label / "missing.json", "w") as f:
            json.dump(missing, f)
        print(f"Saving errors to {output_folder/exp_label}/errors.json")
        with open(output_folder / exp_label / "errors.json", "w") as f:
            json.dump(errors, f)


if __name__ == "__main__":
    main()
