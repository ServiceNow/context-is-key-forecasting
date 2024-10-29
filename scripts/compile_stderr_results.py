"""
Compile the results which contains per-instance "variance" values,
instead of computing the variance using multiple instances.
This reduces the standard errors, since it will no longer take
into account the variation between instances, but only takes
into account the uncertainty from the sampling process.

Note: this script cannot filter the metric results to only keep
the metric from the ROI, since doing so woulud require the
metric computation code to have compute the variance accordingly.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path


logging.basicConfig(level=logging.INFO)

CAP = 5


def get_df(input_folder) -> pd.DataFrame:
    entries = []
    for f in input_folder.glob("*/*/*/*/evaluation"):
        instance = f.parts[-2]
        task = f.parts[-3]
        model = f.parts[-4]
        model_family = f.parts[-5]

        s = open(f, "r").read()
        try:
            entry = eval(s, {"nan": float("nan")})

            entry["model_family"] = model_family
            entry["model"] = model
            entry["Task"] = task
            entry["instance"] = instance

            if entry["metric"] >= CAP:
                entry["metric"] = CAP
                entry["variance"] = 0.0

            entries.append(entry)
        except:
            logging.info(f"Cannot read file for: {model}, {task}, {instance}")

    df = pd.DataFrame(entries)
    if df.variance.isna().any():
        logging.warning(
            f"There are {df.variance.isna().sum()} missing variance entries, "
            + "please regenerate the metrics with CIK_METRIC_COMPUTE_VARIANCE=1 environment variable."
        )

    return df


def get_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    def aggmean_text(x):
        mean = np.mean(x)
        return f"{mean:.3f}"

    def aggvariance_text(x):
        # Formula for the variance of an average
        variance = np.sum(x) / (len(x) ** 2)
        if variance < 0.0:
            # This can happen due to the variance formula not guaranteeing non-negative results,
            # even thought it is unbiased.
            variance = 0.0
        std = variance**0.5
        return f"{std:.3f}"

    df_mean = pd.pivot_table(
        df,
        values="metric",
        index=["Task"],
        columns=["model_family"],
        aggfunc=aggmean_text,
    )
    df_std = pd.pivot_table(
        df,
        values="variance",
        index=["Task"],
        columns=["model_family"],
        aggfunc=aggvariance_text,
    )
    return df_mean + " ± " + df_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/starcaster/data/benchmark/results",
        help="Input folder containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/starcaster/data/benchmark/results",
        help="Output folder for results",
    )

    args = parser.parse_args()
    input_folder = Path(args.input)
    output_folder = Path(args.output)

    df = get_df(input_folder)

    pivot_df = get_pivot_table(df)
    pivot_df = pivot_df.sort_index()
    pivot_df.to_csv(output_folder / f"results_stderr.csv")


if __name__ == "__main__":
    main()
