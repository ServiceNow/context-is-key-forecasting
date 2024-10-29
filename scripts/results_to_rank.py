import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from benchmark import TASK_NAME_TO_WEIGHT, ALL_TASKS


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
)
parser.add_argument(
    "--weight_average",
    action="store_true",
)
args = parser.parse_args()


def find_indices_of_search_string(elements, search_string):
    return [index for index, element in enumerate(elements) if search_string in element]


def extract_mean_std(performance_str):
    try:
        # Split the string at ' ± ' and return the mean and std as floats
        mean, std = performance_str.split(" ± ")
        return float(mean), float(std)
    except:
        return np.nan, np.nan


skill_name_map = {
    "instruction following": "Instruction Following",
    "retrieval: context": "Retrieval: Context",
    "retrieval: memory": "Retrieval: Memory",
    "reasoning: deduction": "Reasoning: Deduction",
    "reasoning: analogy": "Reasoning: Analogy",
    "reasoning: math": "Reasoning: Math",
    "reasoning: causal": "Reasoning: Causal",
}

# Desired skill order
desired_skill_order = [
    "instruction following",
    "retrieval: context",
    "retrieval: memory",
    "reasoning: deduction",
    "reasoning: analogy",
    "reasoning: math",
    "reasoning: causal",
]

NO_CONTEXT_MODELS = [
    "Statsmodels",
    "Lag-Llama",
    "Moirai_large",
    "Chronos_large",
]

desired_context_source_order = ["c_i", "c_h", "c_f", "c_cov", "c_causal"]

# Tasks to ignore
tasks_to_ignore = []
models_to_ignore = ["Naive_random", "Naive_oracle"]

# Track ignored tasks and ignored models
ignored_tasks = []
ignored_models = []

# Read the results csv
data = pd.read_csv(args.input)

# Apply the function to each performance cell, excluding the 'Task' column
performance_means = data.drop(columns=["Task"]).applymap(
    lambda x: extract_mean_std(x)[0]
)
performance_stderrs = data.drop(columns=["Task"]).applymap(
    lambda x: extract_mean_std(x)[1]
)

# Create a copy of the performance_means DataFrame to avoid modifying the original
performance_means_copy = performance_means.copy()
performance_stderrs_copy = performance_stderrs.copy()

# Step 1: Remove tasks where all models have NaN
for index, row in performance_means_copy.iterrows():
    task = data["Task"].iloc[index]

    # If all models for this task have NaN, ignore the task
    if row.isna().all() or task in tasks_to_ignore:
        ignored_tasks.append(task)
        # Drop the task from the DataFrame
        performance_means_copy.drop(index, inplace=True)
        performance_stderrs_copy.drop(index, inplace=True)

# Step 2: Remove models that have NaN in any task or if it is in models_to_ignore
for model in performance_means_copy.columns:
    if performance_means_copy[model].isna().any() or model in models_to_ignore:
        ignored_models.append(model)
        # Drop the model from the DataFrame if it has any NaN
        performance_means_copy.drop(columns=[model], inplace=True)
        performance_stderrs_copy.drop(columns=[model], inplace=True)


models_to_keep = [
    "CC-GPT-4o",
    # 'CC-GPT-4o (no ctx)',
    "CC-GPT-4o-mini",
    # 'CC-GPT-4o-mini (no ctx)',
    "CC-Llama-3.1-405b-instruct-temp10",
    # 'CC-Llama-3.1-405b-instruct-temp10 (no ctx)',
    "CC-OpenRouter-LLaMa-70B-Inst",
    # 'CC-OpenRouter-LLaMa-70B-Inst (no ctx)',
    "CC-OpenRouter-LLaMa-8B-Inst",
    # 'CC-OpenRouter-LLaMa-8B-Inst (no ctx)',
    "CC-OpenRouter-Mixtral-8x7B-Inst",
    # 'CC-OpenRouter-Mixtral-8x7B-Inst (no ctx)',
    # 'Chronos_base',
    "Chronos_large",
    # 'Chronos_mini',
    # 'Chronos_small',
    # 'Chronos_tiny',
    "LLama3-70B",
    # 'LLama3-70B (no ctx)',
    "LLama3-70B-instruct",
    # 'LLama3-70B-instruct (no ctx)',
    "LLama3-8B",
    # 'LLama3-8B (no ctx)',
    "LLama3-8B-instruct",
    # 'LLama3-8B-instruct (no ctx)',
    # 'Lag-Llama',
    "Lag-Llama-GPU",
    "Mixtral-8x7B",
    # 'Mixtral-8x7B (no ctx)',
    "Mixtral-8x7B-Instruct",
    # 'Mixtral-8x7B-Instruct (no ctx)',
    # 'Moirai_base',
    "Moirai_large",
    # 'Moirai_small',
    # 'Naive',
    "Statsmodels",
    # 'TimeLLM-ctx',
    # 'TimeLLM-noctx',
    "UniTime-ctx",
    # 'UniTime-noctx'
]


def simulate_rank(means, std):
    """
    For each model + task pairs, we are given an estimate of the true
    score (means) with an associated standard error (std).
    We then simulate what could be the true score, by assuming that
    the error follows a Gaussian distribution.
    From this simulation, we find the rank of each model, for each task.

    Using this simulation allows us to get an estimate of what the true
    average rank is for each model, with the appropriate standard error.

    Note that this is very crude, but is still better than just
    using the means, or assuming any overlap between the standard errors
    to have equal ranks.
    """
    simulation = means + np.random.normal(loc=0, scale=std)
    rank_df = simulation.apply(lambda row: row.rank(), axis=1)
    return rank_df


def get_weighted_rank_series():
    if args.weight_average:
        weight_series = pd.Series(
            {
                i: float(TASK_NAME_TO_WEIGHT[data["Task"].loc[i]])
                for i in performance_means_copy.index
            }
        )
    else:
        weight_series = pd.Series({i: 1.0 for i in performance_means_copy.index})
    weight_series = weight_series / weight_series.sum()

    # rank_df = simulate_rank(performance_means_copy, performance_stderrs_copy)
    rank_df = simulate_rank(
        performance_means_copy[models_to_keep], performance_stderrs_copy[models_to_keep]
    )
    rank_series = rank_df.mul(weight_series, axis="rows").sum(axis="rows")
    return rank_series


rank_simulations = pd.concat([get_weighted_rank_series() for _ in range(10000)], axis=1)

means = rank_simulations.mean(axis=1)
stderr = rank_simulations.std(axis=1)
df = pd.DataFrame({"means": means, "stderr": stderr}).sort_values("means")
print(
    df.apply(
        lambda row: f'{row["means"]:.3f} $\pm$ {row["stderr"]:.3f}',
        axis=1,
    )
)
