#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantill, Raghav Kansal
"""

import argparse
import os
from math import ceil
from string import Template


def get_fileset(ptype, samples=[]):
    if ptype == "trigger":
        with open("data/SingleMuon_2017.txt", "r") as file:
            filelist = [f[:-1] for f in file.readlines()]

        files = {"2017": filelist}
        fileset = {k: files[k] for k in files.keys()}
        return fileset

    elif ptype == "skimmer":
        from os import listdir

        fileset = {}

        # if "2017_HHToBBVVToBBQQQQ_cHHH1" in samples:
        #     # TODO: replace with UL sample once we have it
        #     with open("data/2017_preUL_nano/HHToBBVVToBBQQQQ_cHHH1.txt", "r") as file:
        #         filelist = [
        #             f[:-1].replace("/eos/uscms/", "root://cmsxrootd.fnal.gov//")
        #             for f in file.readlines()
        #         ]  # need to use xcache redirector at Nebraksa coffea-casa
        #
        #     fileset["2017_HHToBBVVToBBQQQQ_cHHH1"] = filelist[starti:endi]

        if "2017_GluGluToHHTo4V_node_cHHH1" in samples:
            # TODO: replace with UL sample once we have it
            with open("data/2017_preUL_nano/GluGluToHHTo4V_node_cHHH1.txt", "r") as file:
                filelist = [
                    f[:-1].replace("/eos/uscms/", "root://cmsxrootd.fnal.gov//")
                    for f in file.readlines()
                ]  # need to use xcache redirector at Nebraksa coffea-casa

            fileset["2017_GluGluToHHTo4V_node_cHHH1"] = filelist

        if not len(samples) or "2017_HHToBBVVToBBQQQQ_cHHH1" in samples:
            # TODO: replace with UL sample once we have it
            with open("data/2017_preUL_nano/HHToBBVVToBBQQQQ_cHHH1.txt", "r") as file:
                filelist = [
                    f[:-1].replace("/eos/uscms/", "root://cmsxrootd.fnal.gov//")
                    for f in file.readlines()
                ]  # need to use xcache redirector at Nebraksa coffea-casa

            fileset = {"2017_HHToBBVVToBBQQQQ_cHHH1": filelist}

        # extra samples in the folder we don't need for this analysis -
        # TODO: should instead have a list of all samples we need
        ignore_samples = [
            "GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8",
            "GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8",
            "ST_tW_antitop_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
            "ST_tW_top_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
        ]

        for sample in listdir("data/2017_UL_nano/"):
            if sample[-4:] == ".txt" and sample[:-4] not in ignore_samples:
                if not len(samples) or "2017_" + sample[:-4].split("_TuneCP5")[0] in samples:
                    with open(f"data/2017_UL_nano/{sample}", "r") as file:
                        if "JetHT" in sample:
                            filelist = [
                                f[:-1].replace("/hadoop/cms/", "root://redirector.t2.ucsd.edu//")
                                for f in file.readlines()
                            ]
                        else:
                            filelist = [
                                f[:-1].replace("/eos/uscms/", "root://cmsxrootd.fnal.gov//")
                                for f in file.readlines()
                            ]

                    fileset["2017_" + sample[:-4].split("_TuneCP5")[0]] = filelist

        return fileset


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with open(templ_file, "r") as f:
        templ = Template(f.read())

    with open(out_file, "w") as f:
        f.write(templ.substitute(templ_args))


def main(args):
    locdir = "condor/" + args.tag
    homedir = f"/store/user/rkansal/bbVV/{args.processor}/"
    outdir = homedir + args.tag + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    fileset = get_fileset(args.processor, args.samples)

    # directories for every sample
    for sample in fileset:
        os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

    jdl_templ = "src/condor/submit.templ.jdl"
    sh_templ = "src/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        print("Submitting " + sample)

        tot_files = len(fileset[sample])
        njobs = ceil(tot_files / args.files_per_job)

        eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"

        for j in range(njobs):
            if args.test and j == 2:
                break

            localcondor = f"{locdir}/{sample}_{j}.jdl"
            jdl_args = {"dir": locdir, "prefix": sample, "jobid": j}
            write_template(jdl_templ, localcondor, jdl_args)

            localsh = f"{locdir}/{sample}_{j}.sh"
            sh_args = {
                "script": args.script,
                "year": args.year,
                "starti": j * args.files_per_job,
                "endi": (j + 1) * args.files_per_job,
                "sample": sample,
                "processor": args.processor,
                "eosoutpkl": f"{eosoutput_dir}/pickles/out_{j}.pkl",
                "eosoutparquet": f"{eosoutput_dir}/parquet/out_{j}.parquet",
            }
            write_template(sh_templ, localsh, sh_args)
            os.system(f"chmod u+x {localsh}")

            if os.path.exists(f"{localcondor}.log"):
                os.system(f"rm {localcondor}.log")

            print("To submit ", localcondor)

            nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="run.py", help="script to run", type=str)
    parser.add_argument(
        "--test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
        type=bool,
    )
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--tag", dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--processor",
        dest="processor",
        default="trigger",
        help="which processor",
        type=str,
        choices=["trigger", "skimmer"],
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        default=[],
        help="which samples to run, default will be all samples",
        nargs="*",
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    args = parser.parse_args()

    main(args)
