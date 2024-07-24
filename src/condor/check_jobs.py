"""
Checks that there is an output for each job submitted.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
from os import listdir
from pathlib import Path

import numpy as np

from HHbbVV import run_utils
from HHbbVV.run_utils import print_red

parser = argparse.ArgumentParser()
run_utils.parse_common_args(parser)

parser.add_argument("--user", default="rkansal", help="user", type=str)
parser.add_argument("--site", default="lpc", help="t2 site", choices=["lpc", "ucsd"], type=str)
run_utils.add_bool_arg(parser, "check-parquet", default=True, help="check parquet files")
run_utils.add_bool_arg(parser, "submit-missing", default=False, help="submit missing files")
run_utils.add_bool_arg(
    parser,
    "check-running",
    default=False,
    help="check against running jobs as well (running_jobs.txt will be updated automatically)",
)

args = parser.parse_args()

trigger_processor = args.processor.startswith("trigger")

if args.site == "lpc":
    eosdir = f"/eos/uscms/store/user/{args.user}/bbVV/{args.processor}/{args.tag}/{args.year}/"
    user_condor_dir = f"/uscms/home/{args.user}/nobackup/HHbbVV/condor/"
elif args.site == "ucsd":
    eosdir = f"/ceph/cms/store/user/{args.user}/bbVV/{args.processor}/{args.tag}/{args.year}/"
    user_condor_dir = f"/home/users/{args.user}/HHbbVV/condor/"

samples = listdir(eosdir)
jdls = [
    jdl
    for jdl in listdir(f"{user_condor_dir}/{args.processor}/{args.tag}/")
    if jdl.endswith(".jdl")
]

# get the highest numbered .jdl file to know how many output files there should be
jdl_dict = {}
for sample in samples.copy():
    sorted_jdls = np.sort(
        [
            int(jdl[:-4].split("_")[-1])
            for jdl in jdls
            if jdl.split("_")[0] == args.year and "_".join(jdl.split("_")[1:-1]) == sample
        ]
    )

    if len(sorted_jdls):
        jdl_dict[sample] = sorted_jdls[-1] + 1
    else:
        # if for some reason a folder exists in EOS but no .jdl file
        samples.remove(sample)


running_jobs = []
if args.check_running:
    os.system(f"condor_q {args.user} -nobatch" "| awk '{print $9}' > running_jobs.txt")
    with Path("running_jobs.txt").open() as f:
        lines = f.readlines()

    running_jobs = [s[:-4] for s in lines if s.endswith(".sh\n")]


missing_files = []
err_files = []


for sample in samples:
    print(f"Checking {sample}")

    if not trigger_processor and args.check_parquet:
        if not Path(f"{eosdir}/{sample}/parquet").exists():
            print_red(f"No parquet directory for {sample}!")

            for i in range(jdl_dict[sample]):
                if f"{args.year}_{sample}_{i}" in running_jobs:
                    print(f"Job #{i} for sample {sample} is running.")
                    continue

                jdl_file = (
                    f"{user_condor_dir}/{args.processor}/{args.tag}/{args.year}_{sample}_{i}.jdl"
                )
                err_file = f"{user_condor_dir}/{args.processor}/{args.tag}/logs/{args.year}_{sample}_{i}.err"
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

    if not Path(f"{eosdir}/{sample}/pickles").exists():
        print_red(f"No pickles directory for {sample}!")
        continue

    outs_pickles = [
        int(out.split(".")[0].split("_")[-1]) for out in listdir(f"{eosdir}/{sample}/pickles")
    ]

    if trigger_processor or not args.check_parquet:
        print(f"Out pickles: {outs_pickles}")

    for i in range(jdl_dict[sample]):
        if i not in outs_pickles:
            if f"{args.year}_{sample}_{i}" in running_jobs:
                print(f"Job #{i} for sample {sample} is running.")
                continue

            print_red(f"Missing output pickle #{i} for sample {sample}")
            jdl_file = f"{user_condor_dir}/{args.processor}/{args.tag}/{args.year}_{sample}_{i}.jdl"
            err_file = (
                f"{user_condor_dir}/{args.processor}/{args.tag}/logs/{args.year}_{sample}_{i}.err"
            )
            missing_files.append(jdl_file)
            err_files.append(err_file)
            if args.submit_missing:
                os.system(f"condor_submit {jdl_file}")

        if not trigger_processor and args.check_parquet and i not in outs_parquet:
            print_red(f"Missing output parquet #{i} for sample {sample}")


print(f"{len(missing_files)} files to re-run:")
for f in missing_files:
    print(f)

print("\nError files:")
for f in err_files:
    print(f)
