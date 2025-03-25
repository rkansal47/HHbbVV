#!/usr/bin/python

"""
Submits jobs for making the datacards and running fits per sample.

Author(s): Raghav Kansal
"""
from __future__ import annotations

import argparse
import itertools
import os
from math import ceil
from pathlib import Path

from utils import setup

from HHbbVV import run_utils
from HHbbVV.hh_vars import res_sigs as full_samples
from HHbbVV.postprocessing import utils
from HHbbVV.run_utils import add_bool_arg

res_mps = [
    (900, 80),
    (1200, 190),
    (2000, 125),
    (3000, 250),
    (4000, 150),
]

scan_samples = []

for mX, mY in res_mps:
    scan_samples.append(f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mX}_MY-{mY}")

# scan_txbb_wps = ["LP", "MP", "HP"]
# scan_thww_wps = [0.4, 0.6, 0.8, 0.9, 0.94, 0.96, 0.98]

scan_txbb_wps = ["HP"]
scan_thww_wps = [0.6, 0.8]
scan_leadingpt_wps = [300.0, 350.0, 400.0, 450.0]
scan_subleadingpt_wps = [300.0, 350.0, 400.0, 450.0]

nonres_scan_cuts = ["txbb", "bdt"]
res_scan_cuts = ["txbb", "thww", "leadingpt", "subleadingpt"]


def main(args):
    global scan_txbb_wps, scan_thww_wps, scan_leadingpt_wps, scan_subleadingpt_wps  # noqa: PLW0602, PLW0603

    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    local_dir = Path(f"condor/cards/{args.tag}")

    templates_dir = f"/store/user/{username}/bbVV/templates/{args.templates_dir}/"
    cards_dir = f"/store/user/{username}/bbVV/cards/{args.tag}/"

    # make local directory
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # and eos directory
    os.system(f"mkdir -p {t2_local_prefix}/{cards_dir}")

    jdl_templ = f"{submitdir}/submit_cards.templ.jdl"
    sh_templ = f"{submitdir}/submit_cards.templ.sh"

    if args.blinded:
        script = "run_blinded.sh"
    elif args.resonant:
        script = "run_unblinded_res.sh"
    else:
        script = "run_unblinded.sh"

    samples = scan_samples if args.scan else full_samples

    if args.test:
        samples = samples[:2]
        scan_txbb_wps = scan_txbb_wps[-1:]
        scan_thww_wps = scan_thww_wps[-2:]

    scan_wps = list(
        itertools.product(scan_txbb_wps, scan_thww_wps, scan_leadingpt_wps, scan_subleadingpt_wps)
    )
    # remove WPs where subleading pT > leading pT
    scan_wps = [wp for wp in scan_wps if wp[3] <= wp[2]]

    scan_cuts = res_scan_cuts if args.resonant else nonres_scan_cuts

    for sample in samples:
        if args.scan:
            for wps in scan_wps:
                cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)])
                os.system(f"mkdir -p {t2_local_prefix}/{cards_dir}/{cutstr}/{sample}")
        else:
            os.system(f"mkdir -p {t2_local_prefix}/{cards_dir}/{sample}")

    # split along WPs for scan or along # of samples for regular jobs
    njobs = len(scan_wps) if args.scan else ceil(len(samples) / args.files_per_job)

    datacard_args = "--no-do-jshifts" if args.scan else ""

    # submit jobs
    nsubmit = 0

    if args.submit:
        print("Submitting samples")

    for j in range(njobs):
        if args.scan:
            run_samples = samples
            cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, scan_wps[j])])
            run_templates_dir = templates_dir + cutstr
            run_cards_dir = cards_dir + cutstr
        else:
            run_samples = samples[j * args.files_per_job : (j + 1) * args.files_per_job]
            run_templates_dir = templates_dir
            run_cards_dir = cards_dir

        jobid = "-".join(["_".join([str(n) for n in utils.mxmy(s)]) for s in run_samples])

        prefix = "cards" f"{'Scan' if args.scan else ''}"
        local_jdl = Path(f"{local_dir}/{prefix}_{jobid}.jdl")
        local_log = Path(f"{local_dir}/{prefix}_{jobid}.log")
        jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": jobid, "proxy": proxy}
        run_utils.write_template(jdl_templ, local_jdl, jdl_args)

        localsh = f"{local_dir}/{prefix}_{jobid}.sh"
        sh_args = {
            "branch": args.git_branch,
            "gituser": args.git_user,
            "jobnum": jobid,
            "samples": " ".join(run_samples),
            "templates_dir": run_templates_dir,
            "cards_dir": run_cards_dir,
            "datacard_args": datacard_args,
            "script": script,
        }
        run_utils.write_template(sh_templ, localsh, sh_args)
        os.system(f"chmod u+x {localsh}")

        if local_log.exists():
            local_log.unlink()

        if args.submit:
            os.system(f"condor_submit {local_jdl}")
        else:
            print("To submit ", local_jdl)

        nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    parser.add_argument(
        "--git-user", default="rkansal47", help="which user's repo to use", type=str
    )
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument("--templates-dir", help="EOS templates dir", type=str, required=True)
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument("--files-per-job", default=5, help="# samples per condor job", type=int)

    add_bool_arg(parser, "submit", default=False, help="submit files as well as create them")
    add_bool_arg(parser, "resonant", default=True, help="Resonant or nonresonant")
    add_bool_arg(parser, "scan", default=False, help="Scan working points")
    add_bool_arg(parser, "test", default=False, help="run on only 2 samples")
    add_bool_arg(parser, "blinded", default=True, help="blinded or not")

    args = parser.parse_args()

    main(args)
