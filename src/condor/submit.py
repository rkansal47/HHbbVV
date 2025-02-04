#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""
from __future__ import annotations

import argparse
import os
import warnings
from math import ceil
from pathlib import Path

import yaml

from HHbbVV import run_utils

t2_redirectors = {
    "lpc": "root://cmseos.fnal.gov//",
    "ucsd": "root://redirector.t2.ucsd.edu:1095//",
}


def get_proxy(site, username):
    if site == "lpc":
        try:
            proxy = os.environ["X509_USER_PROXY"]
        except:
            print("No valid proxy. Exiting.")
            exit(1)
    elif site == "ucsd":
        if username == "rkansal":
            # Reminder: need to re-copy this from /tmp whenever it expires (symlink?)
            proxy = "/home/users/rkansal/x509up_u31735"
        elif username == "annava":
            proxy = "/home/users/annava/projects/HHbbVV/test"

    return proxy


def main(args):
    run_utils.check_branch(args.git_branch, args.git_user, args.allow_diff_local_repo)

    username = os.environ["USER"]
    proxy = get_proxy(args.site, username)

    if args.site not in args.save_sites:
        warnings.warn(
            f"Your local site {args.site} is not in save sites {args.sites}!", stacklevel=1
        )

    t2_prefixes = [t2_redirectors[site] for site in args.save_sites]

    # t2 output directory
    pdir = Path(f"store/user/{username}/bbbb/{args.processor}/")
    outdir = pdir / args.tag

    # make local directory
    local_dir = Path(f"condor/{args.processor}/{args.tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("CONDOR work dir: ", local_dir)

    fileset = run_utils.get_fileset(
        args.processor,
        args.year,
        args.samples,
        args.subsamples,
        get_num_files=True,
        max_files=args.max_files,
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

            sample_dir = outdir / args.year / subsample
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
                    "t2_prefixes": " ".join(t2_prefixes),
                    "outdir": sample_dir,
                    "jobnum": j,
                    "label": args.label,
                    "njets": args.njets,
                    "save_ak15": "--save-ak15" if args.save_ak15 else "--no-save-ak15",
                    "save_all": "--save-all" if args.save_all else "--no-save-all",
                    "save_skims": "--save-skims" if args.save_skims else "--no-save-skims",
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
    parser.add_argument(
        "--save-sites",
        default=["lpc", "ucsd"],
        help="tier 2s in which we want to save the files",
        type=str,
        nargs="+",
        choices=["lpc", "ucsd"],
    )
    run_utils.add_bool_arg(
        parser,
        "test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    parser.add_argument("--max-files", default=None, help="max total files to run over", type=int)

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

    # YAML check
    if args.yaml is not None:
        with Path(args.yaml).open() as file:
            samples_to_submit = yaml.safe_load(file)

        tag = args.tag
        for key, tdict in samples_to_submit.items():
            for sample, sdict in tdict.items():
                rsample = sample
                if rsample in ["JetHT", "SingleMu"]:
                    rsample += args.year[:4]

                args.samples = [rsample]
                args.subsamples = sdict.get("subsamples", [])
                args.files_per_job = sdict["files_per_job"]
                args.njets = sdict.get("njets", 2)
                args.max_files = sdict.get("max_files", None)
                args.maxchunks = sdict.get("maxchunks", 0)
                args.chunksize = sdict.get("chunksize", 10000)
                args.tag = tag
                args.label = args.jet + sdict["label"] if "label" in sdict else "None"

                if key == "Validation":
                    args.tag = f"{args.tag}_Validation"

                print(args)
                main(args)
    else:
        main(args)
