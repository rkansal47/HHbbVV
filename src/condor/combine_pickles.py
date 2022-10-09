"""
Combines Coffea processor output pickle files

Author(s): Raghav Kansal
"""

import os
from os import listdir
import argparse
import pickle
from coffea.processor.accumulator import accumulate
import sys
from tqdm import tqdm


def accumulate_files(files: list):
    """accumulates pickle files from files list via coffea.processor.accumulator.accumulate"""

    with open(files[0], "rb") as file:
        out = pickle.load(file)

    for ifile in tqdm(files[1:]):
        with open(ifile, "rb") as file:
            out = accumulate([out, pickle.load(file)])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir", default="outfiles", help="directory which contains files to combine", type=str
    )
    parser.add_argument("--name", default="", help="name of combined files", type=str)
    parser.add_argument(
        "--r", default=False, help="combine files in subdirectories of indir", type=bool
    )
    args = parser.parse_args()

    files = [args.indir + "/" + file for file in listdir(args.indir) if file.endswith(".pkl")]

    if args.r:
        dirs = [
            args.indir + "/" + d for d in listdir(args.indir) if os.path.isdir(args.indir + "/" + d)
        ]

        for d in dirs:
            files += [d + "/" + file for file in listdir(d) if file.endswith(".pkl")]

    print(f"Accumulating {len(files)} files")
    out = accumulate_files(files)

    name = args.name if args.name != "" else "combined"

    with open(f"{args.indir}/{args.name}.pkl", "wb") as f:
        pickle.dump(out, f)
