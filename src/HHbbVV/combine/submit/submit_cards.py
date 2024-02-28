#!/usr/bin/python

"""
Submits jobs for making the datacards and running fits per sample.

Author(s): Raghav Kansal
"""

import argparse
import os
from math import ceil
from string import Template
import json
import itertools
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


res_mps = [
    (1000, 100),
    (1000, 125),
    (1000, 150),
    (1000, 190),
    (1000, 250),
    (1000, 60),
    (1000, 80),
    (1200, 100),
    (1200, 150),
    (1200, 190),
    (1200, 250),
    (1200, 60),
    (1200, 80),
    (1400, 100),
    (1400, 125),
    (1400, 150),
    (1400, 190),
    (1400, 250),
    (1400, 60),
    (1400, 80),
    (1600, 100),
    (1600, 125),
    (1600, 150),
    (1600, 190),
    (1600, 250),
    (1600, 60),
    (1600, 80),
    (1800, 100),
    (1800, 125),
    (1800, 150),
    (1800, 190),
    (1800, 250),
    (1800, 60),
    (1800, 80),
    (2000, 100),
    (2000, 125),
    (2000, 150),
    (2000, 190),
    (2000, 250),
    (2000, 60),
    (2000, 80),
    (2200, 100),
    (2200, 125),
    (2200, 150),
    (2200, 190),
    (2200, 250),
    (2200, 60),
    (2200, 80),
    (2400, 100),
    (2400, 125),
    (2400, 150),
    (2400, 190),
    (2400, 250),
    (2400, 60),
    (2400, 80),
    (2600, 100),
    (2600, 125),
    (2600, 150),
    (2600, 190),
    (2600, 250),
    (2600, 80),
    (2800, 100),
    (2800, 125),
    (2800, 150),
    (2800, 190),
    (2800, 250),
    (2800, 60),
    (2800, 80),
    (3000, 100),
    (3000, 125),
    (3000, 150),
    (3000, 190),
    (3000, 250),
    (3000, 60),
    (3000, 80),
    (3500, 100),
    (3500, 125),
    (3500, 150),
    (3500, 190),
    (3500, 250),
    (3500, 60),
    (3500, 80),
    (4000, 100),
    (4000, 125),
    (4000, 150),
    (4000, 190),
    (4000, 250),
    (4000, 60),
    (4000, 80),
    (600, 100),
    (600, 125),
    (600, 150),
    (600, 250),
    (600, 60),
    (600, 80),
    (700, 100),
    (700, 125),
    (700, 150),
    (700, 250),
    (700, 60),
    (700, 80),
    (800, 100),
    (800, 125),
    (800, 150),
    (800, 250),
    (800, 60),
    (800, 80),
    (900, 100),
    (900, 125),
    (900, 150),
    (900, 250),
    (900, 60),
    (900, 80),
]

full_samples = []

for mX, mY in res_mps:
    full_samples.append(f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mX}_MY-{mY}")


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
    global scan_txbb_wps, scan_thww_wps, scan_leadingpt_wps, scan_subleadingpt_wps

    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    local_dir = f"condor/cards/{args.tag}"

    templates_dir = f"/store/user/{username}/bbVV/templates/{args.templates_dir}/"
    cards_dir = f"/store/user/{username}/bbVV/cards/{args.tag}/"

    # make local directory
    logdir = local_dir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and eos directory
    os.system(f"mkdir -p {t2_local_prefix}/{cards_dir}")
    os.system(f"mkdir -p {t2_local_prefix}/{templates_dir}")

    jdl_templ = f"{submitdir}/submit_cards.templ.jdl"
    sh_templ = f"{submitdir}/submit_cards.templ.sh"

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

        prefix = "cards" f"{'Scan' if args.scan else ''}"
        localcondor = f"{local_dir}/{prefix}_{j}.jdl"
        jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
        write_template(jdl_templ, localcondor, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {
            "samples": " ".join(run_samples),
            "in_templates_dir": run_templates_dir,
            "in_cards_dir": run_cards_dir,
            "datacard_args": datacard_args,
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
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument("--templates-dir", help="EOS templates dir", type=str)
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

    args = parser.parse_args()

    main(args)
