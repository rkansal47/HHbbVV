"""
Combines Coffea processor output pickle files

Author(s): Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
import pickle
from os import listdir
from pathlib import Path

from coffea.processor.accumulator import accumulate
from tqdm import tqdm

from HHbbVV import run_utils


def accumulate_files(files: list):
    """accumulates pickle files from files list via coffea.processor.accumulator.accumulate"""

    with Path(files[0]).open("rb") as file:
        out = pickle.load(file)

    for ifile in tqdm(files[1:]):
        with Path(ifile).open("rb") as file:
            out = accumulate([out, pickle.load(file)])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_utils.parse_common_args(parser)  # year, processor, tag
    parser.add_argument("--name", default="combined", help="name of combined files", type=str)
    parser.add_argument(
        "--inuser",
        default=os.getlogin(),
        help="username where pickles are saved (if not you)",
        type=str,
    )
    parser.add_argument(
        "--outuser",
        default=os.getlogin(),
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

    indir = Path(
        f"/eos/uscms/store/user/{args.inuser}/bbVV/{args.processor}/{args.tag}/{args.year}"
    )
    outdir = Path(f"/eos/uscms/store/user/{args.outuser}/bbVV/{args.processor}/{args.tag}/")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Inputs directory:", indir)
    print("Outputs directory:", outdir)

    files = [indir / file for file in listdir(indir) if file.endswith(".pkl")]
    out_dict = {}

    if args.r:
        samples = [d for d in listdir(indir) if (indir / d / "pickles").is_dir()]

        for sample in samples:
            print(sample)
            pickle_path = Path(f"{indir}/{sample}/pickles/")
            sample_files = [
                pickle_path / file for file in listdir(pickle_path) if file.endswith(".pkl")
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

    with (outdir / f"{args.year}_{args.name}.pkl").open("wb") as f:
        pickle.dump(out, f)

    print(f"Saved to {outdir}/{args.year}_{args.name}.pkl")
