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


def main(args):
    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    prefix = f"bias_{args.bias}_seed_{args.seed}"
    local_dir = f"condor/bias/{args.tag}/{prefix}"

    # make local directory
    logdir = local_dir + "/logs"
    os.system(f"mkdir -p {logdir}")

    jdl_templ = f"{submitdir}/submit_bias.templ.jdl"
    sh_templ = f"{submitdir}/submit_bias.templ.sh"

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for j in range(args.num_jobs):
        localcondor = f"{local_dir}/{prefix}_{j}.jdl"
        seed = args.seed + j * args.toys_per_job
        jdl_args = {
            "dir": local_dir,
            "prefix": prefix,
            "jobid": j,
            "proxy": proxy,
            "bias": args.bias,
            "seed": seed,
        }
        write_template(jdl_templ, localcondor, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {
            "seed": seed,
            "num_toys": args.toys_per_job,
            "resonant": "--resonant" if args.resonant else "",
            "bias": args.bias,
            "mintol": args.mintol,
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
    parse_common_args(parser)
    parser.add_argument(
        "--bias", help="expected signal strength to test", type=float, required=True
    )
    parser.add_argument("--mintol", default=0.1, help="minimizer tolerance", type=float)
    args = parser.parse_args()
    main(args)
