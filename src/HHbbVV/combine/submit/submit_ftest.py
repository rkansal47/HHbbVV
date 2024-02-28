#!/usr/bin/python

"""
Splits toy generation into separate condor jobs and fits lowest order + 1 models for F-tests.

Author(s): Raghav Kansal
"""

import argparse
import os
from math import ceil
from string import Template
import json
import sys

from utils import add_bool_arg, write_template, setup, parse_common_args


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
    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    username = os.environ["USER"]
    local_dir = f"condor/f_tests/{args.tag}_{args.low1}{args.low2}"

    # make local directory
    logdir = local_dir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and eos directories
    for i, j in [(0, 0), (0, 1), (1, 0)]:
        os.system(
            f"mkdir -p {t2_local_prefix}//store/user/rkansal/bbVV/cards/f_tests/{args.cards_tag}/"
            f"nTF1_{args.low1 + i}_nTF2_{args.low2 + j}/outs/"
        )

    jdl_templ = f"{submitdir}/submit_ftest.templ.jdl"
    sh_templ = f"{submitdir}/submit_ftest.templ.sh"

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for j in range(args.num_jobs):
        prefix = f"ftests_{args.low1}{args.low2}"
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
