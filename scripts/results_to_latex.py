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
    "--output",
    type=str,
)
parser.add_argument("--weight_average", action="store_true", default=True)
parser.add_argument("--weight_skills", action="store_true", default=True)
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


# Model name map
model_name_map = {
    "CC-GPT-4o": "Direct Prompt - GPT-4o",
    "CC-GPT-4o (no ctx)": "Direct Prompt - GPT-4o (no context)",
    "CC-Llama-3.1-405b-instruct-temp10": "Direct Prompt - Llama-3.1-405B-Instruct",
    "CC-Llama-3.1-405b-instruct-temp10 (no ctx)": "Direct Prompt - Llama-3.1-405B-Instruct (no context)",
    "CC-GPT-4o-mini": "Direct Prompt - GPT-4o-mini",
    "CC-GPT-4o-mini (no ctx)": "Direct Prompt - GPT-4o-mini (no context)",
    "CC-OpenRouter-LLaMa-70B-Inst": "Direct Prompt - LLaMa-70B-Inst",
    "CC-OpenRouter-Mixtral-8x7B-Inst": "Direct Prompt - Mixtral-8x7B-Inst",
    "CC-OpenRouter-LLaMa-8B-Inst": "Direct Prompt - LLaMa-8B-Inst",
    "LLama3-8B": "LLMP - LLama3-8B",
    "LLama3-8B (no ctx)": "LLMP - LLama3-8B (no context)",
    "LLama3-8B-instruct": "LLMP - LLama3-8B-Instruct",
    "LLama3-8B-instruct (no ctx)": "LLMP - LLama3-8B-Instruct (no context)",
    "LLama3-70B": "LLMP - LLama3-70B",
    "LLama3-70B (no ctx)": "LLMP - LLama3-70B (no context)",
    "LLama3-70B-instruct": "LLMP - LLama3-70B-Instruct",
    "LLama3-70B-instruct (no ctx)": "LLMP - LLama3-70B-Instruct (no context)",
    "Mixtral-8x7B": "LLMP - Mixtral-8x7B",
    "Mixtral-8x7B (no ctx)": "LLMP - Mixtral-8x7B (no context)",
    "Mixtral-8x7B-Instruct": "LLMP - Mixtral-8x7B-Instruct",
    "Mixtral-8x7B-Instruct (no ctx)": "LLMP - Mixtral-8x7B-Instruct (no context)",
    "Statsmodels": "Exponential Smoothing (no context)",
    "Lag-Llama": "Lag-Llama (no context)",
}

# LLM models to their associated columns, both with and without context
LLM_MODELS_TO_COLUMNS = {
    "DP GPT-4o": ("CC-GPT-4o", "CC-GPT-4o (no ctx)"),
    "DP Llama-3.1-405B-Inst": (
        "CC-Llama-3.1-405b-instruct-temp10",
        "CC-Llama-3.1-405b-instruct-temp10 (no ctx)",
    ),
    "DP GPT-4o-mini": ("CC-GPT-4o-mini", "CC-GPT-4o-mini (no ctx)"),
    "LLMP Llama3-8B": ("LLama3-8B", "LLama3-8B (no ctx)"),
    "LLMP Llama3-70B": ("LLama3-70B", "LLama3-70B (no ctx)"),
    "LLMP Llama3-8B-Inst": ("LLama3-8B-instruct", "LLama3-8B-instruct (no ctx)"),
    "LLMP Llama3-70B-Inst": (
        "LLama3-70B-instruct",
        "LLama3-70B-instruct (no ctx)",
    ),
    "LLMP Mixtral-8x7B": ("Mixtral-8x7B", "Mixtral-8x7B (no ctx)"),
    "LLMP Mixtral-8x7B-Inst": (
        "Mixtral-8x7B-Instruct",
        "Mixtral-8x7B-Instruct (no ctx)",
    ),
    "DP Llama-70B-Inst": (
        "CC-OpenRouter-LLaMa-70B-Inst",
        "CC-OpenRouter-LLaMa-70B-Inst (no ctx)",
    ),
    "DP Llama-8B-Inst": (
        "CC-OpenRouter-LLaMa-8B-Inst",
        "CC-OpenRouter-LLaMa-8B-Inst (no ctx)",
    ),
    "DP Mixtral-8x7B-Inst": (
        "CC-OpenRouter-Mixtral-8x7B-Inst",
        "CC-OpenRouter-Mixtral-8x7B-Inst (no ctx)",
    ),
    "UniTime": ("UniTime-ctx", "UniTime-noctx"),
    "TimeLLM (ETTh1)": ("TimeLLM-ctx", "TimeLLM-noctx"),
}

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
    "R_Arima",
    "R_ETS",
    "Lag-Llama-GPU",
    "Moirai_large",
    "Chronos_large",
    "TimeGEN1",
]

desired_context_source_order = ["c_i", "c_h", "c_f", "c_cov", "c_causal"]

# Tasks to ignore
tasks_to_ignore = []
models_to_ignore = [
    "Naive_random",
    "Naive_oracle",
    "Naive",
    # Comment from here if you want all models
    # "Moirai_small",
    # "Moirai_base",
    # "Chronos_tiny",
    # "Chronos_mini",
    # "Chronos_small",
    # "Chronos_base",
    # "Mixtral-8x7B (no ctx)",
    # "Mixtral-8x7B-Instruct (no ctx)",
    # "LLama3-8B (no ctx)",
    # "LLama3-8B-instruct (no ctx)",
    # "LLama3-70B (no ctx)",
    # "LLama3-70B-Instruct (no context)",
    # "LLama3-70B-instruct (no ctx)",
    # "CC-Llama-3.1-405b-instruct-temp10 (no ctx)",
    # "CC-GPT-4o-mini (no ctx)",
    # "Direct Prompt - GPT-4o (no ctx)",
    # "CC-OpenRouter-LLaMa-70B-Inst (no ctx)",
    # "CC-OpenRouter-Mixtral-8x7B-Inst (no ctx)",
    # "CC-OpenRouter-LLaMa-8B-Inst (no ctx)",
    # "UniTime-noctx",
    # "TimeLLM-noctx",
]

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

print("Total # of tasks:", len(performance_means_copy))

# Dynamically extract skills and context sources using TaskName._skills and TaskName._context_sources
skills_map = {}
context_sources_map = {}

# To get class
TASKS_STR_TO_TASK = {x.__name__: x for x in ALL_TASKS}

for task in data["Task"]:
    associated_class = TASKS_STR_TO_TASK[task]
    # Access the _skills attribute, excluding "forecasting" and "natural language processing"
    skills_map[task] = [
        s
        for s in associated_class._skills
        if not s in {"forecasting", "natural language processing"}
    ]
    # Access the _context_sources attribute
    context_sources_map[task] = associated_class._context_sources

# For each model, calculate the average performance for each skill and context source

# 1. Create the LaTeX table for skills
skill_data = {}
for model in performance_means_copy.columns:
    entry = {}
    for skill in desired_skill_order:
        total_weight = 0
        total_values = 0.0
        total_variances = 0.0
        for task, task_skills in skills_map.items():
            if (
                skill in task_skills
                and not task in ignored_tasks
                and not task in tasks_to_ignore
            ):
                weight = TASK_NAME_TO_WEIGHT[task] if args.weight_skills else 1
                value = performance_means_copy.loc[data["Task"] == task, model].values[
                    0
                ]
                stderr = performance_stderrs_copy.loc[
                    data["Task"] == task, model
                ].values[0]
                total_weight += weight
                total_values += weight * value
                total_variances += (weight * stderr) ** 2
        if total_weight == 0 and total_values == 0:
            mean = variance = 0
        else:
            mean = total_values / total_weight
            variance = total_variances / (total_weight**2)
        stderr = variance**0.5
        entry[skill] = f"{mean:.3f} ± {stderr:.3f}"

    total_weight = 0
    total_values = 0.0
    total_variances = 0.0
    for task, task_skills in skills_map.items():
        if not task in ignored_tasks and not task in tasks_to_ignore:
            weight = TASK_NAME_TO_WEIGHT[task] if args.weight_average else 1
            value = performance_means_copy.loc[data["Task"] == task, model].values[0]
            stderr = performance_stderrs_copy.loc[data["Task"] == task, model].values[0]
            total_weight += weight
            total_values += weight * value
            total_variances += (weight * stderr) ** 2
    if total_weight == 0 and total_values == 0:
        mean = variance = 0
    else:
        mean = total_values / total_weight
        variance = total_variances / (total_weight**2)
    stderr = variance**0.5
    entry["Average"] = f"{mean:.3f} ± {stderr:.3f}"

    skill_data[model] = entry
skill_df = pd.DataFrame(skill_data).T

print("Total # of models:", len(skill_df))

# Sort the DataFrame by "Average Rank" (best to worst)
skill_df = skill_df.sort_values(by="Average")

# Reorder the columns of the skill DataFrame based on the desired order
skill_df = skill_df[["Average"] + ["Average"] + desired_skill_order]

# Map row and column name
skill_df = skill_df.rename(index=model_name_map)
skill_df = skill_df.rename(columns=skill_name_map)

# Create LaTeX table for skills
latex_skill_table = skill_df.to_latex(index=True, escape=True)
latex_skill_table = latex_skill_table.replace("±", r"$\pm$")

# 2. Create the LaTeX table for context sources
context_data = {}
for model in performance_means_copy.columns:
    entry = {}
    for context in desired_context_source_order:
        total_weight = 0
        total_values = 0.0
        total_variances = 0.0
        for task, task_contexts in context_sources_map.items():
            if (
                context in task_contexts
                and not task in ignored_tasks
                and not task in tasks_to_ignore
            ):
                weight = TASK_NAME_TO_WEIGHT[task] if args.weight_skills else 1
                value = performance_means_copy.loc[data["Task"] == task, model].values[
                    0
                ]
                stderr = performance_stderrs_copy.loc[
                    data["Task"] == task, model
                ].values[0]
                total_weight += weight
                total_values += weight * value
                total_variances += (weight * stderr) ** 2
        if total_weight == 0 and total_values == 0:
            mean = variance = 0
        else:
            mean = total_values / total_weight
            variance = total_variances / (total_weight**2)
        stderr = variance**0.5
        entry[context] = f"{mean:.3f} ± {stderr:.3f}"

    total_weight = 0
    total_values = 0.0
    total_variances = 0.0
    for task, task_contexts in context_sources_map.items():
        if not task in ignored_tasks and not task in tasks_to_ignore:
            weight = TASK_NAME_TO_WEIGHT[task] if args.weight_average else 1
            value = performance_means_copy.loc[data["Task"] == task, model].values[0]
            stderr = performance_stderrs_copy.loc[data["Task"] == task, model].values[0]
            total_weight += weight
            total_values += weight * value
            total_variances += (weight * stderr) ** 2
    if total_weight == 0 and total_values == 0:
        mean = variance = 0
    else:
        mean = total_values / total_weight
        variance = total_variances / (total_weight**2)
    stderr = variance**0.5
    entry["Average"] = f"{mean:.3f} ± {stderr:.3f}"

    context_data[model] = entry
context_df = pd.DataFrame(context_data).T


# Sort the DataFrame by "Average Rank" (best to worst)
context_df = context_df.sort_values(by="Average")

# Reorder the columns of the DataFrame based on the desired order
context_df = context_df[desired_context_source_order]

# Map row and column name
context_df = context_df.rename(index=model_name_map)

# Create LaTeX table for context sources
latex_context_table = context_df.to_latex(index=True, escape=True)
latex_context_table = latex_context_table.replace("±", r"$\pm$")

# Save LaTeX tables to a file on your local machine
with open(args.output, "w") as f:
    f.write(latex_skill_table)
    f.write("\n\n")
    f.write(latex_context_table)

# Step 6: Print ignored tasks and models
print("Saved LaTeX table to", args.output)
print("\nIgnored Tasks:", ignored_tasks)
print("\nIgnored Models:", ignored_models)


# Step 7: Compute for how many tasks the various models "win" against the no-context models
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

wins_to_slots = {
    7: 4,
    6: 3,
    5: 3,
    4: 2,
    3: 2,
    2: 1,
    1: 1,
    0: 0,
}
wins_to_labels = [
    (4, "Beats all"),
    (3, "Beats 5 or 6"),
    (2, "Beats 3 or 4"),
    (1, "Beats 1 or 2"),
    (0, "Beats none"),
]
# wins_to_slots = {
#     7: 4,
#     6: 3,
#     5: 3,
#     4: 3,
#     3: 1,
#     2: 1,
#     1: 1,
#     0: 0,
# }
# wins_to_labels = [
#     (4, "Beats all"),
#     (3, "Beats majority"),
#     (2, None),
#     (1, "Beats minority"),
#     (0, "Beats none"),
# ]
colors = [
    (0, 0.5, 0),
    (1, 1, 0),
    (0.75, 0.75, 1),
    (1, 0.5, 0),
    (0.75, 0, 0),
]
# wins_to_slots = {
#     7: 7,
#     6: 6,
#     5: 5,
#     4: 4,
#     3: 3,
#     2: 2,
#     1: 1,
#     0: 0,
# }
# wins_to_labels = [
#     (7, "7"),
#     (6, "6"),
#     (5, "5"),
#     (4, "4"),
#     (3, "3"),
#     (2, "2"),
#     (1, "1"),
#     (0, "0"),
# ]
# from matplotlib import colormaps
# colors = colormaps["viridis"](np.linspace(0, 1, 8))

all_wins_ctx = [[] for _ in wins_to_labels]
all_wins_no_ctx = [[] for _ in wins_to_labels]
model_names = []

for model, (column_ctx, column_no_ctx) in LLM_MODELS_TO_COLUMNS.items():
    model_names.append(model)
    wins_ctx = [0.0 for _ in wins_to_labels]
    wins_no_ctx = [0.0 for _ in wins_to_labels]
    for index, row in performance_means_copy.iterrows():
        task = data["Task"].loc[index]
        weight = TASK_NAME_TO_WEIGHT[task] if args.weight_average else 1.0

        w_ctx = 0
        w_no_ctx = 0
        for other_model in NO_CONTEXT_MODELS:
            if row[column_ctx] <= row[other_model]:
                w_ctx += 1
            if row[column_no_ctx] <= row[other_model]:
                w_no_ctx += 1
        wins_ctx[wins_to_slots[w_ctx]] += weight
        wins_no_ctx[wins_to_slots[w_no_ctx]] += weight
    for w in range(len(wins_to_labels)):
        all_wins_ctx[w].append(wins_ctx[w])
        all_wins_no_ctx[w].append(wins_no_ctx[w])
all_wins_ctx = [np.array(l) for l in all_wins_ctx]
all_wins_no_ctx = [np.array(l) for l in all_wins_no_ctx]
model_names = np.array(model_names)

# Sort against the number of perfect 4 out of 4 wins tasks
sort_order = np.argsort(all_wins_ctx[-1] + 0.001 * all_wins_ctx[-2])
all_wins_ctx = [l[sort_order] for l in all_wins_ctx]
all_wins_no_ctx = [l[sort_order] for l in all_wins_no_ctx]
model_names = model_names[sort_order]

num_tasks = sum(aw[0] for aw in all_wins_ctx)

fig = plt.figure(figsize=(8, 3.4))
ax1, ax2 = fig.subplots(1, 2, sharey=True)
fig.subplots_adjust(wspace=0.15)

left = np.zeros(len(model_names))
for (wins, label), color in zip(wins_to_labels, colors):
    numbers = all_wins_ctx[wins] / num_tasks
    ax1.barh(
        y=model_names,
        width=numbers,
        label=label,
        left=left,
        color=color,
        height=0.9,
    )
    left += numbers
for x in [0.25, 0.5, 0.75]:
    ax1.axvline(x=x, color="#ffffff", linewidth=0.75)
    ax1.axvline(x=x, color="#3f3f3f", linewidth=0.75, linestyle=":")
ax1.set_title("With context")
ax1.set_xlim([0, 1])
ax1.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
ax1.set_ylim([-0.6, len(model_names) - 1 + 0.6])

left = np.zeros(len(model_names))
for (wins, label), color in zip(wins_to_labels, colors):
    numbers = all_wins_no_ctx[wins] / num_tasks
    ax2.barh(
        y=model_names,
        width=numbers,
        label=label,
        left=left,
        color=color,
        height=0.9,
    )
    left += numbers
for x in [0.25, 0.5, 0.75]:
    ax2.axvline(x=x, color="#ffffff", linewidth=0.75)
    ax2.axvline(x=x, color="#3f3f3f", linewidth=0.75, linestyle=":")
ax2.set_title("Without context")
ax2.set_xlim([0, 1])
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])

ax2.legend(loc="lower right", bbox_to_anchor=(0.95, 1.1), ncols=5)
# fig.tight_layout()
fig.savefig("llm_wins_against_pure_numerical_methods.png", bbox_inches="tight")
fig.savefig(
    "llm_wins_against_pure_numerical_methods.pdf", bbox_inches="tight", transparent=True
)
