#!/usr/bin/python 

import os, sys
import subprocess
import json
import uproot
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import matplotlib.pyplot as plt

import argparse
import warnings

def main(args):

    # read samples to submit
    # TODO: get this to a json that can be identified by year and sample
    with open('data/filelist.txt', 'r') as file:
        filelist = [f[:-1] for f in file.readlines()]
    files = {'2017': filelist}
    fileset = {k: files[k][starti:endi] for k in files.keys()} 

    # files = {}
    # with open('data/fileset_%s_das.json'%(args.year), 'r') as f:
    #     newfiles = json.load(f)
    #     files.update(newfiles)
    # fileset = {k: files[k][starti:endi] for k in args.samples}

    # define processor
    if args.processor=="trigger":
        from HHbbVV import JetHTTriggerEfficienciesProcessor
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
            #ship_env=True,
            #transfer_input_files="HHbbVV",
        )
        cluster.adapt(minimum=1, maximum=30)
        client = Client(cluster)

        exe_args = {
            'client': client,
            'savemetrics': True,
            'schema': BaseSchema, # for base schema
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
        
        outfile = 'outfiles/'+args.year+'.coffea'
        util.save(out, outfile)

    if args.condor:        
        uproot.open.defaults['xrootd_handler'] = uproot.source.xrootd.MultithreadedXRootDSource
            
        exe_args = {'savemetrics':True, 
                    'schema': BaseSchema,
                    #'schema':NanoAODSchema, 
                    'retries': 1}
        
        for filename in infiles:
            print(filename)
            out, metrics = processor.run_uproot_job(
                str(filename), 'Events', p, processor.futures_executor, args, chunksize=10000)
            )
            
            print(f"Output: {out}")
            print(f"Metrics: {metrics}")

            outfile = 'outfiles/'+str(args.year)+'_'+str(args.index)+'.coffea'
            util.save(out, outfile)
        
    return

if __name__ == "__main__":
    # e.g. 
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    parser.add_argument("--processor",  dest="processor",  default="trigger"     help="Trigger processor", type=str)
    parser.add_argument("--dask",       dest="dask",       action="store_true",  default=False, help="Run with dask")
    parser.add_argument("--condor",     dest="condor",     action="store_true",  default=True,  help="Run with condor")
    parser.add_argument('--samples',    dest='samples',    default=[],           help='samples',     nargs='*')
    args = parser.parse_args()

    main(args.year)
