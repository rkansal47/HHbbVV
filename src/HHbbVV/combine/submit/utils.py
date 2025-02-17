from __future__ import annotations

import os
from pathlib import Path

from HHbbVV import run_utils


def setup(args):
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
    submitdir = Path(__file__).resolve().parent

    return t2_local_prefix, t2_prefix, proxy, username, submitdir


def parse_common_args(parser):
    parser.add_argument("--tag", default="Test", help="condor tag", type=str)
    parser.add_argument("--cards-tag", default="Apr26", help="cards dir tag", type=str)
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument("--toys-per-job", default=100, help="# toys per condor job", type=int)
    parser.add_argument("--num-jobs", default=10, help="# condor jobs", type=int)
    parser.add_argument("--seed", default=444, help="# condor jobs", type=int)
    run_utils.add_bool_arg(
        parser, "submit", default=False, help="submit files as well as create them"
    )
    run_utils.add_bool_arg(parser, "resonant", default=True, help="resonant or nonresonant")
    run_utils.add_bool_arg(parser, "blinded", help="Blinded or unblinded?", default=True)
