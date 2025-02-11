#!/usr/bin/python

"""
Splits toy generation into separate condor jobs and fits lowest order + 1 models for F-tests.

Author(s): Raghav Kansal
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils import setup

from HHbbVV import run_utils
from HHbbVV.run_utils import add_bool_arg


def getBestFitR(cards_dir: str):
    """Get best fit r value from the snapshot to set r for toy generation"""
    import ROOT

    f = ROOT.TFile.Open(f"{cards_dir}/higgsCombineSnapshot.MultiDimFit.mH125.root", "READ")
    w = f.Get("w")
    w.loadSnapshot("MultiDimFit")
    return w.var("r").getValV()


def main(args):
    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)
    local_dir = Path(f"condor/f_tests/{args.tag}_{args.low1}{args.low2 if args.resonant else ''}")

    # make local directory
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # and eos directories + best fit rs if needed
    if args.resonant:
        for i, j in [(0, 0), (0, 1), (1, 0)]:
            os.system(
                f"mkdir -p {t2_local_prefix}//store/user/rkansal/bbVV/cards/f_tests/{args.cards_tag}/"
                f"nTF1_{args.low1 + i}_nTF2_{args.low2 + j}/outs/"
            )
    else:
        for i in [0, 1]:
            os.system(
                f"mkdir -p {t2_local_prefix}//store/user/rkansal/bbVV/cards/f_tests/{args.cards_tag}/"
                f"nTF_{args.low1 + i}/outs/"
            )

    if not args.blinded:
        cards_dir = (
            f"nTF_{args.low1}" if not args.resonant else f"nTF1_{args.low1}_nTF2_{args.low2}"
        )
        bestfitr = getBestFitR(cards_dir)
        print("Best Fit r:", bestfitr)
    else:
        bestfitr = None

    jdl_templ = f"{submitdir}/submit_ftest.templ.jdl"
    blind_str = "_unblinded" if not args.blinded else ""
    sh_templ = (
        f"{submitdir}/submit_ftest_res{blind_str}.templ.sh"
        if args.resonant
        else f"{submitdir}/submit_ftest_nonres{blind_str}.templ.sh"
    )

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for j in range(args.num_jobs):
        prefix = f"ftests_{args.low1}{args.low2 if args.resonant else ''}"
        local_jdl = Path(f"{local_dir}/{prefix}_{j}.jdl")
        local_log = Path(f"{local_dir}/{prefix}_{j}.log")

        jdl_args = {
            "dir": local_dir,
            "prefix": prefix,
            "tag": args.cards_tag,
            "jobid": j,
            "proxy": proxy,
        }
        run_utils.write_template(jdl_templ, local_jdl, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {
            "branch": args.git_branch,
            "gituser": args.git_user,
            "jobnum": j,
            "low1": args.low1,
            "low2": args.low2,
            "tag": args.cards_tag,
            "seed": args.seed + j * args.toys_per_job,
            "num_toys": args.toys_per_job,
            "rmax": args.rmax,
            "bestfitr": bestfitr,
        }
        if not args.resonant:
            sh_args.pop("low2")
        if args.blinded:
            sh_args.pop("rmax")
            sh_args.pop("bestfitr")

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
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    parser.add_argument(
        "--git-user", default="rkansal47", help="which user's repo to use", type=str
    )
    parser.add_argument("--tag", default="Test", help="condor tag", type=str)
    parser.add_argument("--cards-tag", help="f tests dir tag", type=str, required=True)
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
    parser.add_argument("--rmax", default=200, help="signal rmax for unblinded fits", type=float)
    add_bool_arg(parser, "submit", default=False, help="submit files as well as create them")
    add_bool_arg(parser, "resonant", default=True, help="nonresonant or resonant")
    add_bool_arg(parser, "blinded", default=True, help="blinded or not")

    args = parser.parse_args()

    main(args)
