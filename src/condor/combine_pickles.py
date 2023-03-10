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


import sys

# needed to import run_utils from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import run_utils


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
    run_utils.parse_common_args(parser)  # year, processor
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--name", default="combined", help="name of combined files", type=str)
    parser.add_argument(
        "--inuser", default="", help="username where pickles are saved (if not you)", type=str
    )
    parser.add_argument(
        "--outuser",
        default="",
        help="username where combined output will be saved (if not you)",
        type=str,
    )
    run_utils.add_bool_arg(
        parser, "r", default=False, help="combine files in sub and subsubdirectories of indir"
    )
    run_utils.add_bool_arg(
        parser,
        "separate-samples",
        default=False,
        help="combine different samples' pickles separately",
    )
    args = parser.parse_args()

    user = os.getlogin()
    if args.inuser == "":
        args.inuser = user
    if args.outuser == "":
        args.outuser = user

    tag_dir = f"/eos/uscms/store/user/{args.inuser}/bbVV/{args.processor}/{args.tag}"
    indir = f"{tag_dir}/{args.year}/"

    outdir = f"/eos/uscms/store/user/{args.outuser}/bbVV/{args.processor}/{args.tag}/"
    os.system(f"mkdir -p {outdir}")

    print(f"Inputs directory:", indir)
    print(f"Outputs directory:", outdir)

    files = [indir + "/" + file for file in listdir(indir) if file.endswith(".pkl")]
    out_dict = {}

    if args.r:
        samples = [d for d in listdir(indir) if os.path.isdir(indir + "/" + d + "/pickles/")]

        for sample in samples:
            print(sample)
            pickle_path = f"{indir}/{sample}/pickles/"
            sample_files = [
                pickle_path + "/" + file for file in listdir(pickle_path) if file.endswith(".pkl")
            ]

            if args.separate_samples:
                out_dict[sample] = accumulate_files(sample_files)
            else:
                files += sample_files

    if args.separate_samples:
        out = {args.year: out_dict}
    else:
        print(f"Accumulating {len(files)} files")
        out = accumulate_files(files)

    with open(f"{outdir}/{args.year}_{args.name}.pkl", "wb") as f:
        pickle.dump(out, f)

    print(f"Saved to {outdir}/{args.year}_{args.name}.pkl")
