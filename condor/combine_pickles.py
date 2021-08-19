"""
Combines Coffea processor output pickle files
Assumes format of output files is {'nevents': int, 'skimmed_events': dict of `column_accumulator`s}, aka output of bbVVSkimmer

Author: Raghav Kansal
"""

import os
from os import listdir
import argparse
import pickle
from coffea.processor.accumulator import accumulate
from coffea.processor import column_accumulator
import sys


def accumulate_files(files: list, norm: bool = False, convert_to_dict: bool = False):
    """ accumulates pickle files from files list via coffea.processor.accumulator's accumulate, divides by nevents at the end if norm is True """

    with open(files[0], 'rb') as file:
        out = pickle.load(file)

    for ifile in files[1:]:
        with open(ifile, 'rb') as file:
            outt = pickle.load(file)
            out = accumulate([out, outt])

    for year, datasets in out.items():
        for dataset, output in datasets.items():
            if norm:
                output['skimmed_events']['weight'] = column_accumulator(output['skimmed_events']['weight'].values / output['nevents'])

            if convert_to_dict:
                output['skimmed_events'] = {
                    key: value.value for (key, value) in output['skimmed_events'].items()
                }

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',      default='outfiles',       help="directory which contains files to combine", type=str)
    parser.add_argument('--year',      default='2017',       help="year", type=str)
    parser.add_argument('--name',      default='',       help="name of combined files", type=str)
    # parser.add_argument('--outdir',      default='',              help="directory in which to place combined files", type=str)
    parser.add_argument('--r',      default=False,              help="combine files in subdirectories of indir", type=bool)
    parser.add_argument('--samples',    default=[],           help='if --r, which samples in indir to combine - if blank (default) will go through all',     nargs='*')
    parser.add_argument('--norm',      default=False,              help="divide weight by total events", type=bool)
    parser.add_argument('--combine-further',      default=True,              help="combine already combined pickles into broader categories ('HHbbVV4q', 'QCD', 'Top', 'V', 'Data')", type=bool)
    args = parser.parse_args()


    if args.r:
        dirs = [args.indir + '/' + dir for dir in listdir(args.indir)]

        if args.name: args.name = '_' + args.name
        acc_str = "_column_accs" if args.combine_further else ""  # to remind that these files have coffea `column_accumulators`, not arrays

        for sample in dirs:
            sample_name = sample.split('/')[-1]

            files = [sample + '/' + file for file in listdir(sample)]
            if not len(files):
                print(f"No files for {sample_name}")
                continue

            print(f"Accumulating {sample_name}")
            accumulated = accumulate_files(files, norm = args.norm and 'JetHT' not in sample_name, convert_to_dict = not args.combine_further)  # normalize if not data, and don't convert to dict if combining further

            with open(f"{args.indir}/{sample_name}{args.name}{acc_str}.pkl", 'wb') as file:
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
        accumulated = accumulate_files(files, norm=args.norm, convert_to_dict=True)

        with open(f"{args.indir}/{args.name}.pkl", 'wb') as file:
            pickle.dump(accumulated, file)

        print(f"Saved as {args.indir}/{args.name}.pkl")


    if args.combine_further:
        os.system(f'mkdir -p {args.indir}/{args.year}_combined{args.name}')
        pickles = [args.indir + '/' + pickle for pickle in listdir(args.indir) if pickle[-4:] == '.pkl']

        combiner = {
            'HHbbVV4q': [pickle for pickle in pickles if f'{args.year}_HHToBBVVToBBQQQQ' in pickle],
            'QCD': [pickle for pickle in pickles if f'{args.year}_QCD' in pickle],
            'Top': [pickle for pickle in pickles if (f'{args.year}_ST' in pickle or f'{args.year}_TT' in pickle)],
            'V': [pickle for pickle in pickles if (f'{args.year}_W' in pickle or f'{args.year}_Z' in pickle)],
            'Data': [pickle for pickle in pickles if f'{args.year}_JetHT' in pickle],
        }

        for key, files in combiner.items():
            print(f"Accumulating {key}: {files}")

            accumulated = accumulate_files(files, norm=False, convert_to_dict=True)
            with open(f"{args.indir}/{args.year}_combined{args.name}/{key}.pkl", 'wb') as file:
                pickle.dump(accumulated, file)

            print(f"Saved as {args.indir}/{args.year}_combined{args.name}/{key}.pkl")
