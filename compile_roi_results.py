import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from benchmark import ALL_TASKS

TASKS_STR_TO_TASK = {x.__name__: x for x in ALL_TASKS}

logging.basicConfig(level=logging.INFO)

CAP = 5

METRIC_NAMES = {
    "standard_crps": "unweighted_CRPS",
    "crps": "RCRPS",
    "roi_crps": "RoI_only_CRPS",
    "non_roi_crps": "nonRoI_only_CRPS",
    "violation_crps": "violation_CRPS",
}


def get_df(input_folder) -> pd.DataFrame:
    entries = []
    for f in input_folder.glob("*/*/*/*/evaluation"):
        instance = f.parts[-2]
        task = f.parts[-3]
        model = f.parts[-4]
        model_family = f.parts[-5]

        if task not in TASKS_STR_TO_TASK.keys():
            continue

        s = open(f, "r").read().replace("nan", "10000000")
        try:
            entry = eval(s)
            entry["model_family"] = model_family
            entry["model"] = model
            entry["Task"] = task
            entry["instance"] = instance

            for key, value in entry.items():
                if type(value) != str and value < 0:
                    entry[key] = 10000000

            entries.append(entry)
        except:
            logging.info(f"Cannot read file for: {model}, {task}, {instance}")
    return pd.DataFrame(entries)


def get_pivot_table(df: pd.DataFrame, metric) -> pd.DataFrame:
    def aggfunc(x):
        x = list(x)
        for idx, value in enumerate(x):
            if value > CAP or np.isnan(value):
                x[idx] = CAP
        if len(x) < 5:
            x.extend([CAP for _ in range(CAP - len(x))])
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        stderr = std / np.sqrt(len(x))
        return f"{mean:.3f} ± {stderr:.3f}"

    def missing_count(x):
        x = list(x)
        return 5 - len(x)

    def count_nans(x):
        x = pd.Series(x)
        return x.isna().sum()

    def get_capped_counts(x):
        x = list(x)
        return len([value for value in x if value > CAP])

    capped_counts = pd.pivot_table(
        df,
        values=metric,
        index=["Task"],
        columns=["model_family"],
        aggfunc=get_capped_counts,
    )
    missing_counts = pd.pivot_table(
        df,
        values=metric,
        index=["Task"],
        columns=["model_family"],
        aggfunc=missing_count,
    )
    nan_counts = pd.pivot_table(
        df, values=metric, index=["Task"], columns=["model_family"], aggfunc=count_nans
    )
    pivot_df = pd.pivot_table(
        df, values=metric, index=["Task"], columns=["model_family"], aggfunc=aggfunc
    )
    return pivot_df, capped_counts, missing_counts, nan_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultsdir",
        type=str,
        default="/starcaster/data/benchmark/results_24_sep_with_transform",
        help="Input folder containing experiment results",
    )

    args = parser.parse_args()
    input_folder = Path(args.resultsdir)
    output_folder = Path(args.resultsdir)

    df = get_df(input_folder)

    significant_failures_count = {}
    missing_failures_count = {}
    nan_failures_count = {}

    # Add the violation_crps to the various CRPS versions
    for metric in METRIC_NAMES.keys():
        df[metric] = df[metric] + df.violation_crps

    for (
        metric,
        output_name,
    ) in METRIC_NAMES.items():
        (
            pivot_df,
            significant_failures_count[metric],
            missing_failures_count[metric],
            nan_failures_count[metric],
        ) = get_pivot_table(df, metric)
        # Get task-level views
        significant_failures_count[metric].to_csv(
            output_folder / f"significant_failures_count_{metric}_df_CAP_{CAP}.csv"
        )
        missing_failures_count[metric].to_csv(
            output_folder / f"missing_failures_count_{metric}_df_CAP_{CAP}.csv"
        )
        nan_failures_count[metric].to_csv(
            output_folder / f"nan_failures_count_{metric}_df_CAP_{CAP}.csv"
        )
        # Aggregate view: sum values from all tasks
        significant_failures_count[metric] = (
            significant_failures_count[metric].sum().to_dict()
        )
        missing_failures_count[metric] = missing_failures_count[metric].sum().to_dict()
        nan_failures_count[metric] = nan_failures_count[metric].sum().to_dict()
        pivot_df = pivot_df.sort_index()
        pivot_df.to_csv(output_folder / f"results_{output_name}_CAP_{CAP}_oct16.csv")

    significant_failures_count_df = pd.DataFrame.from_dict(
        significant_failures_count, orient="index"
    ).astype(int)
    missing_failures_count_df = pd.DataFrame.from_dict(
        missing_failures_count, orient="index"
    ).astype(int)
    nan_failures_count_df = pd.DataFrame.from_dict(
        nan_failures_count, orient="index"
    ).astype(int)
    total_failures_count_df = (
        significant_failures_count_df.add(missing_failures_count_df)
        .add(nan_failures_count_df)
        .astype(int)
    )

    significant_failures_count_df.to_csv(
        output_folder / f"significant_failures_count_df_CAP_{CAP}.csv"
    )
    missing_failures_count_df.to_csv(
        output_folder / f"missing_failures_count_df_CAP_{CAP}.csv"
    )
    nan_failures_count_df.to_csv(output_folder / f"nan_failures_count_df_CAP_{CAP}.csv")
    total_failures_count_df.to_csv(
        output_folder / f"total_failures_count_df_CAP_{CAP}.csv"
    )


if __name__ == "__main__":
    main()
