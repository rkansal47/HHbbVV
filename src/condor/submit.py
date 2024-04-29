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

from HHbbVV import run_utils


def get_site_vars(site):
    username = os.environ["USER"]
    if site == "lpc":
        t2_local_prefix = Path("/eos/uscms/")
        t2_prefix = "root://cmseos.fnal.gov"

        try:
            proxy = os.environ["X509_USER_PROXY"]
        except:
            print("No valid proxy. Exiting.")
            exit(1)
    elif site == "ucsd":
        t2_local_prefix = Path("/ceph/cms/")
        t2_prefix = "root://redirector.t2.ucsd.edu:1095"
        if username == "rkansal":
            proxy = "/home/users/rkansal/x509up_u31735"
        elif username == "annava":
            proxy = "/home/users/annava/projects/HHbbVV/test"

    return username, t2_local_prefix, t2_prefix, proxy


def main(args):
    run_utils.check_branch(args.git_branch, args.git_user, args.allow_diff_local_repo)
    username, t2_local_prefix, t2_prefix, proxy = get_site_vars(args.site)

    homedir = Path(f"store/user/{username}/bbVV/{args.processor}/")
    outdir = homedir / args.tag
    eos_local_dir = t2_local_prefix / outdir
    eos_local_dir.mkdir(parents=True, exist_ok=True)
    print("EOS outputs dir: ", eos_local_dir)

    # make local directory
    local_dir = Path(f"condor/{args.processor}/{args.tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("CONDOR work dir: ", local_dir)

    fileset = run_utils.get_fileset(
        args.processor, args.year, args.samples, args.subsamples, get_num_files=True
    )

    print(f"fileset: {fileset}")

    jdl_templ = "src/condor/submit.templ.jdl"
    sh_templ = "src/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        for subsample, tot_files in fileset[sample].items():
            if args.submit:
                print("Submitting " + subsample)

            (eos_local_dir / args.year / subsample).mkdir(parents=True, exist_ok=True)
            eosoutput_dir = f"{t2_prefix}//{outdir}/{args.year}/{subsample}/"

            njobs = ceil(tot_files / args.files_per_job)

            for j in range(njobs):
                if args.test and j == 2:
                    break

                prefix = f"{args.year}_{subsample}"
                local_jdl = Path(f"{local_dir}/{prefix}_{j}.jdl")
                local_log = Path(f"{local_dir}/{prefix}_{j}.log")
                jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
                run_utils.write_template(jdl_templ, local_jdl, jdl_args)

                localsh = f"{local_dir}/{prefix}_{j}.sh"
                sh_args = {
                    "branch": args.git_branch,
                    "gituser": args.git_user,
                    "script": args.script,
                    "year": args.year,
                    "starti": j * args.files_per_job,
                    "endi": (j + 1) * args.files_per_job,
                    "sample": sample,
                    "subsample": subsample,
                    "processor": args.processor,
                    "maxchunks": args.maxchunks,
                    "chunksize": args.chunksize,
                    "label": args.label,
                    "njets": args.njets,
                    "eosoutpkl": f"{eosoutput_dir}/pickles/out_{j}.pkl",
                    "eosoutparquet": f"{eosoutput_dir}/parquet/out_{j}.parquet",
                    "eosoutroot": f"{eosoutput_dir}/root/nano_skim_{j}.root",
                    "eosoutgithash": f"{eosoutput_dir}/githashes/commithash_{j}.txt",
                    "save_ak15": "--save-ak15" if args.save_ak15 else "--no-save-ak15",
                    "save_all": "--save-all" if args.save_all else "--no-save-all",
                    "lp_sfs": "--lp-sfs" if args.lp_sfs else "--no-lp-sfs",
                    "save_systematics": (
                        "--save-systematics" if args.save_systematics else "--no-save-systematics"
                    ),
                    "inference": "--inference" if args.inference else "--no-inference",
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


def parse_args(parser):
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    parser.add_argument(
        "--git-user", default="rkansal47", help="which user's repo to use", type=str
    )
    parser.add_argument("--script", default="src/run.py", help="script to run", type=str)
    parser.add_argument("--outdir", default="outfiles", help="directory for output files", type=str)
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    run_utils.add_bool_arg(
        parser,
        "test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)

    run_utils.add_bool_arg(
        parser, "submit", default=False, help="submit files as well as create them"
    )

    run_utils.add_bool_arg(
        parser,
        "allow-diff-local-repo",
        default=False,
        help="Allow the local repo to be different from the specified remote repo (not recommended!)."
        "If false, submit script will exit if the latest commits locally and on Github are different.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_utils.parse_common_args(parser)
    parse_args(parser)
    args = parser.parse_args()
    main(args)
