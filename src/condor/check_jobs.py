"""
Checks that there is an output for each job submitted.

Author: Raghav Kansal
"""

import os
from os import listdir
import sys
import numpy as np
import argparse


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
args = parser.parse_args()


eosdir = f"/eos/uscms/store/user/rkansal/bbVV/{args.processor}/{args.tag}/{args.year}/"

samples = listdir(eosdir)
jdls = [jdl for jdl in listdir(f"condor/{args.processor}/{args.tag}/") if jdl.endswith(".jdl")]

jdl_dict = {
    sample: np.sort(
        [int(jdl[:-4].split("_")[-1]) for jdl in jdls if "_".join(jdl.split("_")[1:-1]) == sample]
    )[-1]
    for sample in samples
}

for sample in samples:
    if args.processor != "trigger":
        outs_parquet = [
            int(out.split(".")[0].split("_")[-1]) for out in listdir(f"{eosdir}/{sample}/parquet")
        ]
        print(outs_parquet)

    outs_pickles = [
        int(out.split(".")[0].split("_")[-1]) for out in listdir(f"{eosdir}/{sample}/pickles")
    ]
    print(f"Checking {sample}")

    for i in range(jdl_dict[sample]):
        if i not in outs_pickles:
            print(f"Missing output pickle #{i} for sample {sample}")
            # os.system(f"condor_submit condor/{tag}/2017_{sample}_{i}.jdl")

        if args.processor != "trigger":
            if i not in outs_parquet:
                print(f"Missing output parquet #{i} for sample {sample}")
