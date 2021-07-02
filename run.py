#!/usr/bin/python

import json
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor
import pickle

import argparse
import warnings


def main(args):

    # read samples to submit
    # TODO: get this to a json that can be identified by year and sample
    with open('data/filelist.txt', 'r') as file:
        filelist = [f[:-1] for f in file.readlines()]
    files = {'2017': filelist}
    fileset = {k: files[k][args.starti:args.endi] for k in files.keys()}

    # files = {}
    # with open('data/fileset_%s_das.json'%(args.year), 'r') as f:
    #     newfiles = json.load(f)
    #     files.update(newfiles)
    # fileset = {k: files[k][starti:endi] for k in args.samples}

    # define processor
    if args.processor == "trigger":
        from processors import JetHTTriggerEfficienciesProcessor
        # TODO: add year as argument to processor
        p = JetHTTriggerEfficienciesProcessor()
    else:
        warnings.warn('Warning: no processor declared')
        return

    if args.dask:
        import time
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        tic = time.time()
        cluster = LPCCondorCluster(
            # ship_env=True,
            # transfer_input_files="HHbbVV",
        )
        cluster.adapt(minimum=1, maximum=30)
        client = Client(cluster)

        exe_args = {
            'client': client,
            'savemetrics': True,
            'schema': BaseSchema,  # for base schema
            # 'schema': nanoevents.NanoAODSchema, # for nano schema in the future
            'align_clusters': True,
        }

        print("Waiting for at least one worker...")
        client.wait_for_workers(1)

        out, metrics = processor.run_uproot_job(
            fileset,
            treename="Events",
            processor_instance=p,
            executor=processor.dask_executor,
            executor_args=exe_args,
            #    maxchunks=10
        )

        elapsed = time.time() - tic
        print(f"num: {out['num'].view(flow=True)}")
        print(f"den: {out['den'].view(flow=True)}")
        print(f"Metrics: {metrics}")
        print(f"Finished in {elapsed:.1f}s")

        filehandler = open('out.hist', 'wb')
        pickle.dump(out, filehandler)
        filehandler.close()


    if args.condor:
        uproot.open.defaults['xrootd_handler'] = uproot.source.xrootd.MultithreadedXRootDSource

        exe_args = {'savemetrics':True,
                    'schema': BaseSchema,
                    # 'schema':NanoAODSchema,
                    'retries': 1}

        out, metrics = processor.run_uproot_job(
            fileset,
            'Events',
            p,
            processor.futures_executor,
            exe_args,
            chunksize=10000,
    #        maxchunks=1
        )

        print(f"Output: {out}")
        print(f"Metrics: {metrics}")

        filehandler = open(f'outfiles/{args.year}_{args.starti}-{args.endi}.hist', 'wb')
        pickle.dump(out, filehandler)
        filehandler.close()


    return


if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    # parser.add_argument('--outdir',     dest='outdir',     default='outfiles',   help="directory for output files", type=str)
    parser.add_argument("--processor",  dest="processor",  default="trigger",    help="Trigger processor", type=str)
    parser.add_argument("--dask",       dest="dask",       action="store_true",  default=False, help="Run with dask")
    parser.add_argument("--condor",     dest="condor",     action="store_true",  default=True,  help="Run with condor")
    parser.add_argument('--samples',    dest='samples',    default=[],           help='samples',     nargs='*')
    args = parser.parse_args()

    main(args)
