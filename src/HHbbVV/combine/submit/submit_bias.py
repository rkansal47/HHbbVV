#!/usr/bin/python

"""
Splits toy generation into separate condor jobs and fits lowest order + 1 models for F-tests.

Author(s): Raghav Kansal
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils import parse_common_args, setup

from HHbbVV import run_utils


def main(args):
    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    prefix = f"bias_{args.bias}_seed_{args.seed}"
    local_dir = Path(f"condor/bias/{args.tag}/{prefix}")

    # make local directory
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    jdl_templ = f"{submitdir}/submit_bias.templ.jdl"
    sh_templ = f"{submitdir}/submit_bias.templ.sh"

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for j in range(args.num_jobs):
        local_jdl = Path(f"{local_dir}/{prefix}_{j}.jdl")
        local_log = Path(f"{local_dir}/{prefix}_{j}.log")

        seed = args.seed + j * args.toys_per_job
        jdl_args = {
            "dir": local_dir,
            "prefix": prefix,
            "jobid": j,
            "proxy": proxy,
            "bias": args.bias,
            "seed": seed,
        }
        run_utils.write_template(jdl_templ, local_jdl, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {
            "seed": seed,
            "num_toys": args.toys_per_job,
            "resonant": "--resonant" if args.resonant else "",
            "bias": args.bias,
            "mintol": args.mintol,
        }
        run_utils.write_template(sh_templ, localsh, sh_args)
        os.system(f"chmod u+x {localsh}")

        if local_log.exists():
            local_log.unlink()

        if args.submit:
            os.system(f"condor_submit {local_jdl}")
        else:
            print("To submit ", local_jdl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    parser.add_argument(
        "--bias", help="expected signal strength to test", type=float, required=True
    )
    parser.add_argument("--mintol", default=0.1, help="minimizer tolerance", type=float)
    args = parser.parse_args()
    main(args)
