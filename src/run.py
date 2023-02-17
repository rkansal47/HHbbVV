#!/usr/bin/python

"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

import pickle
import os
import argparse

import numpy as np
import uproot

from coffea import nanoevents
from coffea import processor

import run_utils


def run_dask(p: processor, fileset: dict, args):
    """Run processor on using dask via lpcjobqueue"""
    import time
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster

    tic = time.time()
    cluster = LPCCondorCluster(
        ship_env=True,
        shared_temp_directory="/tmp",
        transfer_input_files="HHbbVV",
    )
    client = Client(cluster)
    nanoevents_plugin = run_utils.NanoeventsSchemaPlugin()  # update nanoevents schema
    client.register_worker_plugin(nanoevents_plugin)
    cluster.adapt(minimum=1, maximum=30)

    print("Waiting for at least one worker")
    client.wait_for_workers(1)

    # does treereduction help?
    executor = processor.DaskExecutor(status=True, client=client)
    run = processor.Runner(
        executor=executor,
        savemetrics=True,
        schema=nanoevents.PFNanoAODSchema,
        chunksize=args.chunksize,
    )
    hists, metrics = run(
        {key: fileset[key] for key in args.samples}, "Events", processor_instance=p
    )

    elapsed = time.time() - tic
    print(f"hists: {hists}")
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")

    with open("hists.pkl", "wb") as f:
        pickle.dump(hists, f)


def run(p: processor, fileset: dict, args):
    """Run processor without fancy dask (outputs then need to be accumulated manually)"""
    run_utils.add_mixins(nanoevents)  # update nanoevents schema

    # outputs are saved here as pickles
    outdir = "./outfiles"
    os.system(f"mkdir -p {outdir}")

    if args.processor in ["skimmer", "input", "ttsfs"]:
        # these processors store intermediate files in the "./outparquet" local directory
        local_dir = os.path.abspath(".")
        local_parquet_dir = os.path.abspath(os.path.join(".", "outparquet"))

        if os.path.isdir(local_parquet_dir):
            os.system(f"rm -rf {local_parquet_dir}")

        os.system(f"mkdir {local_parquet_dir}")

    uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

    if args.executor == "futures":
        executor = processor.FuturesExecutor(status=True)
    else:
        executor = processor.IterativeExecutor(status=True)

    run = processor.Runner(
        executor=executor,
        savemetrics=True,
        schema=nanoevents.PFNanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=None if args.maxchunks == 0 else args.maxchunks,
    )

    out, metrics = run(fileset, "Events", processor_instance=p)

    # print(out)

    filehandler = open(f"{outdir}/{args.starti}-{args.endi}.pkl", "wb")
    pickle.dump(out, filehandler)
    filehandler.close()

    # need to combine all the files from these processors before transferring to EOS
    # otherwise it will complain about too many small files
    if args.processor in ["skimmer", "input", "ttsfs"]:
        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow as pa

        pddf = pd.read_parquet(local_parquet_dir)

        if args.processor in ["skimmer", "ttsfs"]:
            # need to write with pyarrow as pd.to_parquet doesn't support different types in
            # multi-index column names
            table = pa.Table.from_pandas(pddf)
            pq.write_table(table, f"{local_dir}/{args.starti}-{args.endi}.parquet")

        if args.processor == "input":
            # save as root files for input skimmer

            import awkward as ak

            with uproot.recreate(
                f"{local_dir}/nano_skim_{args.starti}-{args.endi}.root",
                compression=uproot.LZ4(4),
            ) as rfile:
                rfile["Events"] = ak.Array(
                    # take only top-level column names in multiindex df
                    {key: np.squeeze(pddf[key].values) for key in pddf.columns.levels[0]}
                )


def main(args):
    p = run_utils.get_processor(args.processor, args.save_ak15, args.label, args.njets)

    if len(args.files):
        fileset = {f"{args.year}_{args.files_name}": args.files}
    else:
        fileset = run_utils.get_fileset(
            args.processor, args.year, args.samples, args.subsamples, args.starti, args.endi
        )

    print(f"Running on fileset {fileset}")

    if args.executor == "dask":
        run_dask(p, fileset, args)
    else:
        run(p, fileset, args)


if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="year", type=str, required=True, choices=["2016APV", "2016", "2017", "2018"])
    parser.add_argument("--starti", default=0, help="start index of files", type=int)
    parser.add_argument("--endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--processor",
        default="trigger",
        help="Trigger processor",
        type=str,
        choices=["trigger", "skimmer", "input", "ttsfs"],
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
    parser.add_argument(
        "--files", default=[], help="set of files to run on instead of samples", nargs="*"
    )
    parser.add_argument(
        "--files-name",
        type=str,
        default="files",
        help="sample name of files being run on, if --files option used",
    )
    parser.add_argument("--chunksize", type=int, default=10000, help="chunk size in processor")
    parser.add_argument("--label", default="AK8_H_VV", help="label", type=str)
    parser.add_argument("--njets", default=2, help="njets", type=int)
    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)

    run_utils.add_bool_arg(
        parser, "save-ak15", default=False, help="run inference for and save ak15 jets"
    )

    args = parser.parse_args()

    main(args)
