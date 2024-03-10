#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""
from __future__ import annotations

import argparse
import os
from math import ceil
from pathlib import Path

import submit

from HHbbVV import run_utils

samples = [
    # "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1000_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1400_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-1800_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2200_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2400_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-190",
    # "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2600_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-2800_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-150",
    # "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-60",
    # "NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-190",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-700_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-700_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-700_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-700_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-700_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-700_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-800_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-800_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-800_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-800_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-800_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-800_MY-80",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-100",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-125",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-150",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-250",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-60",
    "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80",
]


def main(args):
    username, t2_local_prefix, t2_prefix, proxy = submit.get_site_vars(args.site)
    run_utils.check_branch(args.git_branch, args.allow_diff_local_repo)

    local_dir = f"condor/postprocessing/{args.tag}"
    homedir = f"/store/user/{username}/bbVV/templates/"
    outdir = homedir + args.tag + "/"

    # make local directory
    logdir = local_dir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p {t2_local_prefix}/{outdir}")

    # and eos directory
    eosoutput_dir = f"{t2_prefix}/{outdir}/"
    (t2_local_prefix / outdir).mkdir(parents=True, exist_ok=True)

    jdl_templ = "src/condor/submit_pp.templ.jdl"
    sh_templ = "src/condor/submit_pp.templ.sh"

    # submit jobs
    nsubmit = 0
    print("Submitting samples")

    njobs = ceil(len(samples) / args.files_per_job)

    for j in range(njobs):
        run_samples = " ".join(samples[j * args.files_per_job : (j + 1) * args.files_per_job])

        prefix = "templates"

        local_jdl = Path(f"{local_dir}/{prefix}_{j}.jdl")
        local_log = Path(f"{local_dir}/{prefix}_{j}.log")

        jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
        run_utils.write_template(jdl_templ, local_jdl, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {
            "branch": args.git_branch,
            "samples": run_samples,
            "eosout": eosoutput_dir,
            "eosoutgithash": f"{eosoutput_dir}/githashes/commithash_{j}.txt",
        }
        run_utils.write_template(sh_templ, localsh, sh_args)
        os.system(f"chmod u+x {localsh}")

        if local_log.exists():
            local_log.unlink()

        print("To submit ", local_jdl)
        if args.submit:
            os.system(f"condor_submit {local_jdl}")

        nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    submit.parse_args(parser)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    args = parser.parse_args()
    main(args)
