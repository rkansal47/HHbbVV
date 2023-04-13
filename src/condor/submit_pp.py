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

# needed to import run_utils from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import run_utils


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
    os.system(f"mkdir -p {t2_local_prefix}/{outdir}/")

    jdl_templ = "src/condor/submit_pp.templ.jdl"
    sh_templ = "src/condor/submit_pp.templ.sh"

    # submit jobs
    nsubmit = 0
    print("Submitting samples")

    njobs = ceil(len(samples) / args.files_per_job)

    for j in range(njobs):
        run_samples = " ".join(samples[j * args.files_per_job : (j + 1) * args.files_per_job])

        prefix = "templates"
        localcondor = f"{local_dir}/{prefix}_{j}.jdl"
        jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
        write_template(jdl_templ, localcondor, jdl_args)

        localsh = f"{local_dir}/{prefix}_{j}.sh"
        sh_args = {"samples": run_samples, "eosout": eosoutput_dir}
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
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument("--files-per-job", default=5, help="# files per condor job", type=int)

    run_utils.add_bool_arg(
        parser, "submit", default=False, help="submit files as well as create them"
    )

    args = parser.parse_args()

    main(args)
