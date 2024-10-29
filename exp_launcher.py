"""
Launches all experiment in the `experiments` folder.

Json files prefixed with "_" are ignored.

"""

import argparse
import glob
import json
import os
import subprocess
import tenacity

from datetime import datetime


CURRENT_TIME = str(int(datetime.now().timestamp()))
STARCASTER_DATA_OBJECT = "snow.research.starcaster.data"


LAUNCH_COMMAND = """
eai job new \
    --name {label}_starcaster\
    --cpu {n_cpu}\
    --gpu {n_gpu}\
    --mem {cpu_mem}\
    --image registry.toolkit-sp.yul201.service-now.com/snow.research.starcaster/interactive_toolkit\
    --data {STARCASTER_DATA_OBJECT}:/starcaster/data\
    --env CUDA_LAUNCH_BLOCKING=1\
    --env CIK_METRIC_COMPUTE_VARIANCE=1\
    --preemptable\
    -- bash -c 'git config --global user.email "results@starcaster.ai" && git config --global user.name "StarCaster Result Uploader" && conda init && source /tmp/.bashrc && source /starcaster/data/benchmark/configs && conda activate /starcaster/data/benchmark/conda && huggingface-cli login --token $HF_TOKEN && cd {code_path} && pip install -r requirements.txt && python run_baselines.py --exp-spec {exp_spec} --output /starcaster/data/benchmark/{resultsdir} {other_args}'
"""


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    before=lambda retry_state: retry_state.attempt_number > 1
    and print(f"Retrying... attempt {retry_state.attempt_number} of 5"),
)
def push_code():
    """
    Push the local code to the cluster.
    """
    code_dir = f"starcaster_code_{CURRENT_TIME}"

    print("Pushing code to toolkit...")
    cmd = [
        "bash",
        "-c",
        f"rsync -a . /tmp/{code_dir} --delete --exclude-from='.eaiignore' && "
        f"eai data push {STARCASTER_DATA_OBJECT} /tmp/{code_dir}:./benchmark/code/{code_dir} && "
        f"rm -rf /tmp/{code_dir}",
    ]

    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if "bad request error" in [result.stdout.lower(), result.stderr.lower()]:
        print("Error pushing code to toolkit. Try running the launcher again.")
        exit(1)

    return f"/starcaster/data/benchmark/code/{code_dir}"


def launch_all(dir, resultsdir):
    # Push all content of current directory (and subdirectories)
    code_dir = push_code()

    # Launch all experiments
    # XXX: Skips all json files starting with "_"
    for exp_spec in glob.glob(f"{dir}/[!_]*.json"):
        launch_job(exp_spec, code_dir, resultsdir)


def launch_job(exp_spec, code_dir, resultsdir, other_args=""):
    if exp_spec.startswith("./"):
        exp_spec = exp_spec[2:]
    job_spec = exp_spec.split(".")[0]

    label = (
        (exp_spec.split(".")[0].split("/")[-1] + f"_{CURRENT_TIME}")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    use_gpu = "g" in job_spec.split("_")[-1]
    n_resources = int(job_spec.split("_")[-1].replace("g", "").replace("c", ""))
    command = LAUNCH_COMMAND.format(
        # Job specification
        label=label,
        exp_spec=exp_spec,
        code_path=code_dir,
        resultsdir=resultsdir,
        # Compute resources
        n_cpu=(n_resources * 2) if use_gpu else n_resources,
        n_gpu=n_resources if use_gpu else 0,
        cpu_mem=(80 * n_resources) if use_gpu else 32,
        # Other
        STARCASTER_DATA_OBJECT=STARCASTER_DATA_OBJECT,
        other_args=other_args,
    )
    print(command)
    os.system(command)


def create_summary_json(dir, summary_file):
    # Load all experiments from the directory
    exps = []
    for exp_spec in glob.glob(f"{dir}/[!_]*.json"):
        with open(exp_spec, "r") as f:
            exp = json.load(f)
            if isinstance(exp, list):
                exps += exp
            else:
                exps.append(exp)

    # Sort by experiment label
    exps = sorted(exps, key=lambda x: x["label"])

    # Save summary file
    with open(os.path.join(dir, summary_file), "w") as f:
        json.dump(exps, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir",
        type=str,
        help="Directory containing experiment files to launch",
        default="experiments",
    )
    parser.add_argument(
        "--killall",
        action="store_true",
        help="Kill all running jobs",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Launch a job that summarizes all experiments and uploads to GitHub",
    )
    parser.add_argument(
        "--resultsdir", type=str, default="results_24_sep_with_transform"
    )
    parser.add_argument(
        "--cap",
        type=float,
        help="Cap value to cap each instance's metric",
    )

    args = parser.parse_args()

    if args.killall:
        print("Killing all running jobs...")
        os.system(
            "eai job ls --me --state alive --fields id,name | grep starcaster | cut -d ' ' -f1 | xargs eai job kill || echo 'Nothing to kill.'"
        )
        exit(0)

    if args.summary:
        print("Launching summary job...")
        summary_file = "_summary_c40.json"
        create_summary_json(args.expdir, summary_file)

        # Ask the user if they agree
        exps = json.load(open(os.path.join(args.expdir, summary_file), "r"))
        print("The following experiments will be summarized:")
        for exp in exps:
            print(f"  - {exp['label']}")

        confirm = input("Do you want to launch the summary job? y/n:")
        if confirm.lower() != "y":
            print("Aborting.")
            exit(0)

        other_args = "--skip-cache-miss  --upload-results"
        if args.cap:
            other_args = (
                other_args + f" --cap {args.cap}"
            )  # --cap is only necessary in the summary mode, as it would be better to have per-model result CSVs without the cap, for our reference
        launch_job(
            exp_spec=os.path.join(args.expdir, summary_file),  # 40 CPUs
            code_dir=push_code(),
            resultsdir=args.resultsdir,
            other_args=other_args,
        )
        exit(0)

    launch_all(args.expdir, args.resultsdir)
