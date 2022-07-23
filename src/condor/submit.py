#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

import argparse
import os
from math import ceil
from string import Template
import json


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


def get_fileset(processor, year, samples, subsamples):
    with open(f"data/pfnanoindex_{year}.json", "r") as f:
        full_fileset = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset[year][sample]
        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

        # if so keep only that subset
        if len(get_subsamples):
            sample_set = {subsample: sample_set[subsample] for subsample in get_subsamples}

        sample_set = {
            subsample: ["root://cmsxrootd.fnal.gov//" + fname for fname in sample_set[subsample]]
            for subsample in sample_set
        }

        fileset[sample] = sample_set

    return fileset


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with open(templ_file, "r") as f:
        templ = Template(f.read())

    with open(out_file, "w") as f:
        f.write(templ.substitute(templ_args))


def main(args):
    try:
        proxy = os.environ["X509_USER_PROXY"]
    except:
        print("No valid proxy. Exiting.")
        exit(1)

    username = os.environ["USER"]
    locdir = "condor/" + args.tag
    homedir = f"/store/user/{username}/bbVV/{args.processor}/"
    outdir = homedir + args.tag + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    fileset = get_fileset(args.processor, args.year, args.samples, args.subsamples)

    jdl_templ = "src/condor/submit.templ.jdl"
    sh_templ = "src/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        for subsample in fileset[sample]:
            print("Submitting " + subsample)
            os.system(f"mkdir -p /eos/uscms/{outdir}/{args.year}/{subsample}")

            tot_files = len(fileset[sample][subsample])
            njobs = ceil(tot_files / args.files_per_job)

            eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{args.year}/{subsample}/"

            for j in range(njobs):
                if args.test and j == 2:
                    break

                prefix = f"{args.year}_{subsample}"
                localcondor = f"{locdir}/{prefix}_{j}.jdl"
                jdl_args = {"dir": locdir, "prefix": prefix, "jobid": j, "proxy": proxy}
                write_template(jdl_templ, localcondor, jdl_args)

                localsh = f"{locdir}/{prefix}_{j}.sh"
                sh_args = {
                    "script": args.script,
                    "year": args.year,
                    "starti": j * args.files_per_job,
                    "endi": (j + 1) * args.files_per_job,
                    "sample": sample,
                    "subsample": subsample,
                    "processor": args.processor,
                    "maxchunks": args.maxchunks,
                    "label": args.label,
                    "njets": args.njets,
                    "eosoutpkl": f"{eosoutput_dir}/pickles/out_{j}.pkl",
                    "eosoutparquet": f"{eosoutput_dir}/parquet/out_{j}.parquet",
                    "eosoutroot": f"{eosoutput_dir}/root/nano_skim_{j}.root",
                    "save_ak15": "--save-ak15" if args.save_ak15 else "--no-save-ak15",
                }
                write_template(sh_templ, localsh, sh_args)
                os.system(f"chmod u+x {localsh}")

                if os.path.exists(f"{localcondor}.log"):
                    os.system(f"rm {localcondor}.log")

                print("To submit ", localcondor)
                if args.submit:
                    os.system("condor_submit %s" % localcondor)
                nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="run.py", help="script to run", type=str)
    parser.add_argument(
        "--test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
        type=bool,
    )
    parser.add_argument("--year", default="2017", help="year", type=str)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--processor",
        default="trigger",
        help="which processor",
        type=str,
        choices=["trigger", "skimmer", "input"],
    )
    parser.add_argument(
        "--samples",
        default=[],
        help="which samples to run",  # , default will be all samples",
        nargs="*",
    )
    parser.add_argument(
        "--subsamples",
        default=[],
        help="which subsamples, by default will be all in the specified sample(s)",
        nargs="*",
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)
    parser.add_argument("--label", default="AK15_H_VV", help="label", type=str)
    parser.add_argument("--njets", default=2, help="njets", type=int)
    parser.add_argument(
        "--submit", dest="submit", action="store_true", help="submit jobs when created"
    )

    add_bool_arg(parser, "save-ak15", default=False)

    args = parser.parse_args()

    main(args)
