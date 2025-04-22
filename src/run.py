#!/usr/bin/python

"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import uproot
from coffea import nanoevents, processor

from HHbbVV import run_utils


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

    with Path("hists.pkl").open("wb") as f:
        pickle.dump(hists, f)


def run(p: processor, fileset: dict, args):
    """Run processor without fancy dask (outputs then need to be accumulated manually)"""
    run_utils.add_mixins(nanoevents)  # update nanoevents schema

    # outputs are saved here as pickles
    outdir = Path("outfiles")
    outdir.mkdir(exist_ok=True)

    if args.processor in ["skimmer", "input", "ttsfs"]:
        # these processors store intermediate files in the "./outparquet" local directory
        local_dir = Path().resolve()
        local_parquet_dir = local_dir / "outparquet"

        if local_parquet_dir.is_dir():
            os.system(f"rm -rf {local_parquet_dir}")

        local_parquet_dir.mkdir()

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

    for i in range(5):
        try:
            out, metrics = run(fileset, "Events", processor_instance=p)
            break
        except OSError as e:
            import time

            if i < 4:
                print(f"Caught OSError: {e}")
                print("Retrying in 1 minute...")
                time.sleep(60)
            else:
                raise e

    with (outdir / f"{args.starti}-{args.endi}.pkl").open("wb") as f:
        pickle.dump(out, f)

    print(out)

    # need to combine all the files from these processors before transferring to EOS
    # otherwise it will complain about too many small files
    if args.processor in ["skimmer", "input", "ttsfs"] and args.save_skims:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        pddf = pd.read_parquet(local_parquet_dir)

        if args.processor in ["skimmer", "ttsfs"]:
            # need to write with pyarrow as pd.to_parquet doesn't support different types in
            # multi-index column names
            print(list(pddf.columns))
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
    p = run_utils.get_processor(
        args.processor,
        args.save_ak15,
        args.label,
        args.njets,
        args.save_systematics,
        args.inference,
        args.save_all,
        args.save_skims,
        args.lp_sfs,
        args.jme_presel,
    )

    if len(args.files):
        fileset = {f"{args.year}_{args.files_name}": args.files}
    else:
        fileset = run_utils.get_fileset(
            args.processor, args.year, args.samples, args.subsamples, args.starti, args.endi
        )

    ignore_files = [
        "root://cmseos.fnal.gov///store/user/lpcpfnano/rkansal/v2_3/2016APV/XHY/NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-80_TuneCP5_13TeV-madgraph-pythia8/NMSSM_XToYHTo2W2BTo4Q2B_MX-600_MY-80/230323_175525/0000/nano_mc2016pre_36.root",
        "root://cmseos.fnal.gov///store/user/lpcpfnano/rkansal/v2_3/2016/XHY/NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-190_TuneCP5_13TeV-madgraph-pythia8/NMSSM_XToYHTo2W2BTo4Q2B_MX-1600_MY-190/230323_195705/0000/nano_mc2016post_27.root",
        "'root://cmseos.fnal.gov///store/user/lpcpfnano/rkansal/v2_3/2016APV/XHY/NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-80_TuneCP5_13TeV-madgraph-pythia8/NMSSM_XToYHTo2W2BTo4Q2B_MX-3500_MY-80/230323_175525/0000/nano_mc2016pre_16.root",
    ]

    for key in fileset:
        for file in ignore_files:
            if file in fileset[key]:
                fileset[key].remove(file)

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
    run_utils.parse_common_args(parser)
    parser.add_argument("--starti", default=0, help="start index of files", type=int)
    parser.add_argument("--endi", default=-1, help="end index of files", type=int)
    parser.add_argument(
        "--executor",
        type=str,
        default="iterative",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument(
        "--files", default=[], help="set of files to run on instead of samples", nargs="*"
    )
    parser.add_argument(
        "--files-name",
        type=str,
        default="files",
        help="sample name of files being run on, if --files option used",
    )

    args = parser.parse_args()

    main(args)
