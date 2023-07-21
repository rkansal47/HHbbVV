#!/usr/bin/python

"""
Splits toy generation into separate condor jobs and fits lowest order + 1 models for F-tests.

Author(s): Raghav Kansal
"""

import argparse
import os
from math import ceil
from string import Template
import json

import sys

from utils import add_bool_arg, write_template, setup, parse_common_args

import ROOT


def _tolist(argset):
    return [x.GetName() for x in argset]


def getParameters():
    """Get nuisance parameters from workspace"""
    f = ROOT.TFile.Open("combined_withmasks.root", "READ")
    w = f.Get("w")
    ps = _tolist(w.allVars())
    pois = _tolist(w.set("ModelConfig_POI"))
    obs = _tolist(w.genobj("ModelConfig").GetObservables())

    ret_ps = []
    for p in ps:
        if not (
            "mcstat" in p
            or "qcdparam" in p
            or p.endswith("_In")
            or p.endswith("__norm")
            or p.startswith("n_exp_")
            or p.startswith("mask_")
            or p in pois
            or p in obs
        ):
            ret_ps.append(p)

    return ret_ps


def main(args):
    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    prefix = "impacts"
    local_dir = f"condor/impacts/{args.tag}/"

    # make local directory
    logdir = local_dir + "/logs"
    os.system(f"mkdir -p {logdir}")

    jdl_templ = f"{submitdir}/submit_impacts.templ.jdl"
    sh_templ = f"{submitdir}/submit_impacts.templ.sh"

    ps = getParameters()
    print(f"Running impacts on {len(ps)} parameters:")
    print(*ps, sep="\n")

    res_str = "-r" if args.resonant else ""
    commands = [f"run_blinded.sh {res_str} -i"] + [
        f"run_blinded.sh {res_str} --impactsf {p}" for p in ps
    ]
    impactfiles = ["higgsCombine_initialFit_impacts.MultiDimFit.mH125.root"] + [
        f"higgsCombine_paramFit_impacts_{p}.MultiDimFit.mH125.root" for p in ps
    ]

    collect_command = f"run_blinded.sh {res_str} --impactsc {','.join(ps)}"

    if args.local:
        print("Running locally")
        for command in commands:
            os.system(command)
        os.system(collect_command)
        return

    collect_sh = f"{local_dir}/collect.sh"
    with open(collect_sh, "w") as f:
        f.write(collect_command)

    os.system(f"chmod u+x {collect_sh}")
    print(f"To collect impacts afterwards, run: {collect_sh}")

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for p, command, impactfile in zip(["init"] + ps, commands, impactfiles):
        localcondor = f"{local_dir}/{prefix}_{p}.jdl"
        jdl_args = {
            "dir": local_dir,
            "prefix": prefix,
            "jobid": p,
            "proxy": proxy,
            "impactfile": impactfile,
        }
        write_template(jdl_templ, localcondor, jdl_args)

        localsh = f"{local_dir}/{prefix}_{p}.sh"
        sh_args = {"command": "./" + command}
        write_template(sh_templ, localsh, sh_args)
        os.system(f"chmod u+x {localsh}")

        if os.path.exists(f"{localcondor}.log"):
            os.system(f"rm {localcondor}.log")

        if args.submit:
            os.system("condor_submit %s" % localcondor)
        else:
            print("To submit ", localcondor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    add_bool_arg(parser, "local", help="run locally", default=False)
    args = parser.parse_args()
    main(args)
