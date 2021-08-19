"""
Combines Coffea processor output pickle files
Assumes format of output files is {'nevents': int, 'skimmed_events': dict of `column_accumulator`s}, aka output of bbVVSkimmer

Author: Raghav Kansal
"""

from os import listdir
import argparse
import pickle
from coffea.processor.accumulator import accumulate
from coffea.processor import column_accumulator
import sys


def accumulate_files(files: list, norm: bool = False):
    """ accumulates pickle files from files list via coffea.processor.accumulator's accumulate, divides by nevents at the end if norm is True """

    with open(files[0], 'rb') as file:
        out = pickle.load(file)

    for ifile in files[1:]:
        with open(ifile, 'rb') as file:
            outt = pickle.load(file)
            out = accumulate([out, outt])

    out['skimmed_events'] = {
        key: value.value for (key, value) in out['skimmed_events'].items()
    }

    if norm:  # and 'JetHT' not in sample:
        out['skimmed_events']['weight'] /= out['nevents']

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',      default='outfiles',       help="directory which contains files to combine", type=str)
    parser.add_argument('--name',      default='all',       help="name of combined files", type=str)
    # parser.add_argument('--outdir',      default='',              help="directory in which to place combined files", type=str)
    parser.add_argument('--r',      default=False,              help="combine files in subdirectories of indir", type=bool)
    parser.add_argument('--samples',    default=[],           help='if --r, which samples in indir to combine - if blank (default) will go through all',     nargs='*')
    parser.add_argument('--norm',      default=False,              help="divide weight by total events", type=bool)
    args = parser.parse_args()

    if args.r:
        dirs = [args.indir + '/' + dir for dir in listdir(args.indir)]
        for sample in dirs:
            sample_name = sample.split('/')[-1]

            files = [sample + '/' + file for file in listdir(sample)]
            if not len(files):
                print(f"No files for {sample_name}")
                continue

            print(f"Accumulating {sample_name}")
            accumulated = accumulate_files(files, norm = args.norm and 'JetHT' not in sample_name)  # normalize if not data

            if args.name: args.name = '_' + args.name
            with open(f"{args.indir}/{sample_name}{args.name}.pkl", 'wb') as file:
                pickle.dump(accumulated, file)

            print(f"Saved as {args.indir}/{sample_name}{args.name}.pkl")

    else:
        if not args.name:
            print("Name can't be blank -- exiting")
            sys.exit()

        files = [args.indir + '/' + file for file in listdir(args.indir)]
        if not len(files):
            print("No files to accumulate -- exiting")
            sys.exit()

        print(f"Accumulating {args.indir}")
        accumulated = accumulate_files(files, norm=args.norm)

        with open(f"{args.indir}/{args.name}.pkl", 'wb') as file:
            pickle.dump(accumulated, file)

        print(f"Saved as {args.indir}/{args.name}.pkl")
