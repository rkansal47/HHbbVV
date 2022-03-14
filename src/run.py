#!/usr/bin/python

"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantill, Raghav Kansal
"""

import numpy as np

import uproot
from coffea.nanoevents import BaseSchema
from coffea import nanoevents
from coffea import processor
import pickle
import os
import json

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


def get_fileset(processor, year, samples, subsamples, starti, endi):
    with open(f"data/pfnanoindex_{year}.json", "r") as f:
        full_fileset = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset[year][sample]
        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

        # if so keep only that subset
        if len(get_subsamples):
            sample_set = {subsample: sample_set[subsample] for subsample in get_subsamples}

        sample_set = {
            f"{year}_{subsample}": [
                "root://cmsxrootd.fnal.gov//" + fname
                for fname in sample_set[subsample][starti:endi]
            ]
            for subsample in sample_set
        }

        fileset = {**fileset, **sample_set}

    return fileset

def get_xsecs():
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

        p = bbVVSkimmer(xsecs=get_xsecs())
    elif args.processor == "input":
        from HHbbVV.processors import TaggerInputSkimmer
        p = TaggerInputSkimmer.TaggerInputSkimmer(args.label)

    fileset = get_fileset(
        args.processor, args.year, args.samples, args.subsamples, args.starti, args.endi
    )

    print(fileset)

    if args.executor == "dask":
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
            executor=executor,
            savemetrics=True,
            schema=nanoevents.NanoAODSchema,
            chunksize=args.chunksize,
        )
        out, metrics = run(
            {key: fileset[key] for key in args.samples}, "Events", processor_instance=p
        )

        elapsed = time.time() - tic
        print(f"Metrics: {metrics}")
        print(f"Finished in {elapsed:.1f}s")
    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        executor = (
            processor.futures_executor
            if args.executor == "futures"
            else processor.iterative_executor
        )

        exe_args = {
            "savemetrics": True,
            "schema": nanoevents.NanoAODSchema,
        }

        out, metrics = processor.run_uproot_job(
            fileset,
            treename="Events",
            processor_instance=p,
            executor=executor,
            executor_args=exe_args,
            chunksize=args.chunksize,
        )

        filehandler = open(f"outfiles/{args.starti}-{args.endi}.pkl", "wb")
        pickle.dump(out, filehandler)
        filehandler.close()

        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow as pa

        print("reading parquet")

        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        pddf = pd.read_parquet(local_dir)

        print("read parquet")

        os.system(f"mkdir -p {local_dir}")
        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{os.path.abspath('.')}/{args.starti}-{args.endi}.parquet")

        print("dumped parquet")


if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2017", help="year", type=str)
    parser.add_argument("--starti", default=0, help="start index of files", type=int)
    parser.add_argument("--endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--processor",
        default="trigger",
        help="Trigger processor",
        type=str,
        choices=["trigger", "skimmer", "input"],
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="iterative",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument("--samples", default=[], help="samples", nargs="*")
    parser.add_argument("--subsamples", default=[], help="subsamples", nargs="*")
    parser.add_argument("--chunksize", type=int, default=10000, help="chunk size in processor")
    parser.add_argument("--label", default="AK15_H_VV", help="label", type=str)
    args = parser.parse_args()

    main(args)
