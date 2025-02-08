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

import submit

from HHbbVV import run_utils
from HHbbVV.hh_vars import res_sigs as samples


def main(args):
    run_utils.check_branch(args.git_branch, args.git_user, args.allow_diff_local_repo)

    # username, t2_local_prefix, t2_prefix, proxy = submit.get_site_vars(args.site)
    username = os.environ["USER"]
    proxy = submit.get_proxy(args.site, username)

    if args.site not in args.save_sites:
        warnings.warn(
            f"Your local site {args.site} is not in save sites {args.sites}!", stacklevel=1
        )

    t2_prefixes = [submit.t2_redirectors[site] for site in args.save_sites]

    # t2 output directory
    pdir = Path(f"store/user/{username}/bbVV/templates/")
    outdir = pdir / args.tag

    # make local directory
    local_dir = Path(f"condor/postprocessing/{args.tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("Condor work dir: ", local_dir)

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
            "gituser": args.git_user,
            "t2_prefixes": " ".join(t2_prefixes),
            "outdir": outdir,
            "jobnum": j,
            "samples": run_samples,
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
