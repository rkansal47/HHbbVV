"""
Checks that there is an output for each job submitted.

Author: Raghav Kansal
"""

import os
from os import listdir
from os.path import exists
import sys
import numpy as np
import argparse
from colorama import Fore, Style

# needed to import run_utils from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import run_utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "--processor",
    default="trigger",
    help="which processor",
    type=str,
    choices=["trigger", "skimmer", "input", "ttsfs"],
)

parser.add_argument("--tag", default="", help="tag for jobs", type=str)
parser.add_argument("--year", default="2017", help="year", type=str)
parser.add_argument("--user", default="rkansal", help="user", type=str)
run_utils.add_bool_arg(parser, "submit-missing", default=False, help="submit missing files")
run_utils.add_bool_arg(
    parser,
    "check-running",
    default=False,
    help="check against running jobs as well (running_jobs.txt will be updated automatically)",
)

args = parser.parse_args()


eosdir = f"/eos/uscms/store/user/{args.user}/bbVV/{args.processor}/{args.tag}/{args.year}/"

samples = listdir(eosdir)
jdls = [
    jdl
    for jdl in listdir(
        f"/uscms/home/{args.user}/nobackup/HHbbVV/condor/{args.processor}/{args.tag}/"
    )
    if jdl.endswith(".jdl")
]

jdl_dict = {
    sample: np.sort(
        [
            int(jdl[:-4].split("_")[-1])
            for jdl in jdls
            if jdl.split("_")[0] == args.year and "_".join(jdl.split("_")[1:-1]) == sample
        ]
    )[-1]
    + 1
    for sample in samples
}


def print_red(s):
    return print(f"{Fore.RED}{s}{Style.RESET_ALL}")


running_jobs = []
if args.check_running:
    os.system("condor_q | awk '{print $9}' > running_jobs.txt")
    with open("running_jobs.txt", "r") as f:
        lines = f.readlines()

    running_jobs = [s[:-4] for s in lines if s.endswith(".sh\n")]


missing_files = []
err_files = []


for sample in samples:
    print(f"Checking {sample}")

    if args.processor != "trigger":
        if not exists(f"{eosdir}/{sample}/parquet"):
            print_red(f"No parquet directory for {sample}!")

            for i in range(jdl_dict[sample]):
                if f"{args.year}_{sample}_{i}" in running_jobs:
                    print(f"Job #{i} for sample {sample} is running.")
                    continue

                jdl_file = f"condor/{args.processor}/{args.tag}/{args.year}_{sample}_{i}.jdl"
                err_file = f"condor/{args.processor}/{args.tag}/logs/{args.year}_{sample}_{i}.err"
                print(jdl_file)
                missing_files.append(jdl_file)
                err_files.append(err_file)
                if args.submit_missing:
                    os.system(f"condor_submit {jdl_file}")

            continue

        outs_parquet = [
            int(out.split(".")[0].split("_")[-1]) for out in listdir(f"{eosdir}/{sample}/parquet")
        ]
        print(f"Out parquets: {outs_parquet}")

    if not exists(f"{eosdir}/{sample}/pickles"):
        print_red(f"No pickles directory for {sample}!")
        continue

    outs_pickles = [
        int(out.split(".")[0].split("_")[-1]) for out in listdir(f"{eosdir}/{sample}/pickles")
    ]

    if args.processor == "trigger":
        print(f"Out pickles: {outs_pickles}")

    for i in range(jdl_dict[sample]):
        if i not in outs_pickles:
            if f"{args.year}_{sample}_{i}" in running_jobs:
                print(f"Job #{i} for sample {sample} is running.")
                continue

            print_red(f"Missing output pickle #{i} for sample {sample}")
            jdl_file = f"condor/{args.processor}/{args.tag}/{args.year}_{sample}_{i}.jdl"
            err_file = f"condor/{args.processor}/{args.tag}/logs/{args.year}_{sample}_{i}.err"
            missing_files.append(jdl_file)
            err_files.append(err_file)
            if args.submit_missing:
                os.system(f"condor_submit {jdl_file}")

        if args.processor != "trigger":
            if i not in outs_parquet:
                print_red(f"Missing output parquet #{i} for sample {sample}")


print(f"{len(missing_files)} files to re-run:")
for f in missing_files:
    print(f)

print(f"\nError files:")
for f in err_files:
    print(f)
