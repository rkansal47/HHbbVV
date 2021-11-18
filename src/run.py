#!/usr/bin/python

"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantill, Raghav Kansal
"""

import uproot
from coffea.nanoevents import BaseSchema
from coffea import nanoevents
from coffea import processor
import pickle
import os

import argparse

import warnings

from distributed.diagnostics.plugin import WorkerPlugin


def fxn():
    warnings.warn("userwarning", UserWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

nanoevents.NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
    "FatJetAK15_nConstituents",
    "JetPFCandsAK15",
)
nanoevents.NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
nanoevents.NanoAODSchema.mixins["PFCands"] = "PFCand"


class NanoeventsSchemaPlugin(WorkerPlugin):
    def __init__(self):
        pass

    def setup(self, worker):
        from coffea import nanoevents

        nanoevents.NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
            "FatJetAK15_nConstituents",
            "JetPFCandsAK15",
        )
        nanoevents.NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
        nanoevents.NanoAODSchema.mixins["PFCands"] = "PFCand"


def get_fileset(ptype, samples, starti, endi):
    if ptype == "trigger":
        with open("data/SingleMuon_2017.txt", "r") as file:
            filelist = [f[:-1] for f in file.readlines()]

        files = {"2017": filelist}
        fileset = {k: files[k][starti:endi] for k in files.keys()}
        return fileset

    elif ptype == "skimmer":
        from os import listdir

        fileset = {}

        if "2017_HHToBBVVToBBQQQQ_cHHH1" in samples:
            # TODO: replace with UL sample once we have it
            with open("data/2017_preUL_nano/HHToBBVVToBBQQQQ_cHHH1.txt", "r") as file:
                filelist = [
                    f[:-1].replace("/eos/uscms/", "root://cmsxrootd.fnal.gov//")
                    for f in file.readlines()
                ]  # need to use xcache redirector at Nebraksa coffea-casa

            fileset["2017_HHToBBVVToBBQQQQ_cHHH1"] = filelist[starti:endi]

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
                if "2017_" + sample[:-4].split("_TuneCP5")[0] in samples:
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

                    fileset["2017_" + sample[:-4].split("_TuneCP5")[0]] = filelist[starti:endi]

        return fileset


def get_xsecs():
    import json

    with open("data/xsecs.json") as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def main(args):

    # define processor
    if args.processor == "trigger":
        from HHbbVV.processors import JetHTTriggerEfficienciesProcessor

        p = JetHTTriggerEfficienciesProcessor()
    elif args.processor == "skimmer":
        from HHbbVV.processors import bbVVSkimmer

        xsecs = get_xsecs()
        p = bbVVSkimmer(
            xsecs=xsecs,
            condor=args.condor,
            output_location=args.outdir,
        )
        # p = bbVVSkimmer(xsecs=xsecs, condor=args.condor, output_location=os.getcwd())

    fileset = get_fileset(args.processor, args.samples, args.starti, args.endi)

    if args.condor:
        print("condor")
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        # executor = processor.FuturesExecutor(compression=1, status=True)
        executor = processor.IterativeExecutor(status=True)
        run = processor.Runner(
            executor=executor, savemetrics=True, schema=nanoevents.NanoAODSchema
        )  # chunksize=10000
        out, metrics = run(
            {key: fileset[key] for key in args.samples}, "Events", processor_instance=p
        )

    elif args.dask:
        print("dask")
        import time
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        tic = time.time()
        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files="src/HHbbVV",
        )
        client = Client(cluster)
        nanoevents_plugin = NanoeventsSchemaPlugin()
        client.register_worker_plugin(nanoevents_plugin)
        cluster.adapt(minimum=1, maximum=30)

        print("Waiting for at least one worker")
        client.wait_for_workers(1)

        # does treereduction help?
        executor = processor.DaskExecutor(status=True, client=client, treereduction=2)
        run = processor.Runner(
            executor=executor, savemetrics=True, schema=nanoevents.NanoAODSchema, chunksize=100000
        )
        out, metrics = run(
            {key: fileset[key] for key in args.samples}, "Events", processor_instance=p
        )

        elapsed = time.time() - tic
        print(f"Metrics: {metrics}")
        print(f"Finished in {elapsed:.1f}s")


if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--starti", dest="starti", default=0, help="start index of files", type=int)
    parser.add_argument("--endi", dest="endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--processor",
        dest="processor",
        default="trigger",
        help="Trigger processor",
        type=str,
        choices=["trigger", "skimmer"],
    )
    parser.add_argument(
        "--dask", dest="dask", action="store_true", default=False, help="Run with dask"
    )
    parser.add_argument(
        "--condor", dest="condor", action="store_true", default=False, help="Run with condor"
    )
    parser.add_argument("--samples", dest="samples", default=[], help="samples", nargs="*")

    args = parser.parse_args()

    main(args)
