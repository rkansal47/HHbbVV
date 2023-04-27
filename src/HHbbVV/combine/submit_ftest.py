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

import sys


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


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with open(templ_file, "r") as f:
        templ = Template(f.read())

    with open(out_file, "w") as f:
        f.write(
            templ.safe_substitute(
                templ_args,
            )
        )


def main(args):
    if args.site == "lpc":
        t2_local_prefix = "/eos/uscms/"
        t2_prefix = "root://cmseos.fnal.gov"

        try:
            proxy = os.environ["X509_USER_PROXY"]
        except:
            print("No valid proxy. Exiting.")
            exit(1)
    elif args.site == "ucsd":
        t2_local_prefix = "/ceph/cms/"
        t2_prefix = "root://redirector.t2.ucsd.edu:1095"
        proxy = "/home/users/rkansal/x509up_u31735"

    username = os.environ["USER"]
    local_dir = f"condor/f_tests/{args.tag}"

    # make local directory
    logdir = local_dir + "/logs"
    os.system(f"mkdir -p {logdir}")

    jdl_templ = "src/HHbbVV/combine/submit_ftest.templ.jdl"
    sh_templ = "src/HHbbVV/combine/resonant_ftest_templ.sh"

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for j in range(args.num_jobs):
        prefix = "ftests"
        localcondor = f"{local_dir}/{prefix}_{j}.jdl"
        jdl_args = {
            "dir": local_dir,
            "prefix": prefix,
            "tag": args.cards_tag,
            "jobid": j,
            "proxy": proxy,
        }
        write_template(jdl_templ, localcondor, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {
            "in_low1": args.low1,
            "in_low2": args.low2,
            "in_tag": args.cards_tag,
            "in_seed": args.seed + j * args.toys_per_job,
            "in_num_toys": args.toys_per_job,
        }
        write_template(sh_templ, localsh, sh_args)
        os.system(f"chmod u+x {localsh}")

        if os.path.exists(f"{localcondor}.log"):
            os.system(f"rm {localcondor}.log")

        if args.submit:
            os.system("condor_submit %s" % localcondor)
        else:
            print("To submit ", localcondor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="Test", help="condor tag", type=str)
    parser.add_argument("--cards-tag", default="Apr26", help="f tests dir tag", type=str)
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument("--low1", default=0, help="low order poly in dim 1", type=int)
    parser.add_argument("--low2", default=0, help="low order poly in dim 2", type=int)
    parser.add_argument("--toys-per-job", default=100, help="# toys per condor job", type=int)
    parser.add_argument("--num-jobs", default=10, help="# condor jobs", type=int)
    parser.add_argument("--seed", default=444, help="# condor jobs", type=int)
    add_bool_arg(parser, "submit", default=False, help="submit files as well as create them")

    args = parser.parse_args()

    main(args)
