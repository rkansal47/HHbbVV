#!/usr/bin/python

"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantill, Raghav Kansal
"""

import uproot
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea import processor
import pickle

import argparse
import warnings


def get_fileset(ptype, samples, starti, endi):
    if ptype == 'trigger':
        with open('data/SingleMuon_2017.txt', 'r') as file:
            filelist = [f[:-1] for f in file.readlines()]

        files = {'2017': filelist}
        fileset = {k: files[k][starti:endi] for k in files.keys()}
        return fileset

    elif ptype == 'skimmer':
        from os import listdir

        fileset = {}

        if '2017_HHToBBVVToBBQQQQ_cHHH1' in samples:
            # TODO: replace with UL sample once we have it
            with open('data/2017_preUL_nano/HHToBBVVToBBQQQQ_cHHH1.txt', 'r') as file:
                filelist = [f[:-1].replace('/eos/uscms/', 'root://cmsxrootd.fnal.gov//') for f in file.readlines()]   # need to use xcache redirector at Nebraksa coffea-casa

            fileset['2017_HHToBBVVToBBQQQQ_cHHH1'] = filelist[starti:endi]

        # extra samples in the folder we don't need for this analysis - TODO: should instead have a list of all samples we need
        ignore_samples = ['GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8',
                          'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
                          'ST_tW_antitop_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8',
                          'ST_tW_top_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8']

        for sample in listdir('data/2017_UL_nano/'):
            if sample[-4:] == '.txt' and sample[:-4] not in ignore_samples:
                if '2017_' + sample[:-4].split('_TuneCP5')[0] in samples:
                    with open(f'data/2017_UL_nano/{sample}', 'r') as file:
                        if 'JetHT' in sample: filelist = [f[:-1].replace('/hadoop/cms/', 'root://redirector.t2.ucsd.edu//') for f in file.readlines()]
                        else: filelist = [f[:-1].replace('/eos/uscms/', 'root://cmsxrootd.fnal.gov//') for f in file.readlines()]

                    fileset['2017_' + sample[:-4].split('_TuneCP5')[0]] = filelist[starti:endi]

        return fileset


def get_xsecs():
    import json
    with open('data/xsecs.json') as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def main(args):

    # define processor
    if args.processor == "trigger":
        from processors import JetHTTriggerEfficienciesProcessor
        p = JetHTTriggerEfficienciesProcessor()
    elif args.processor == 'skimmer':
        from processors import bbVVSkimmer
        xsecs = get_xsecs()
        p = bbVVSkimmer(xsecs=xsecs, condor=args.condor)

    fileset = get_fileset(args.processor, args.samples, args.starti, args.endi)

    if args.condor:
        uproot.open.defaults['xrootd_handler'] = uproot.source.xrootd.MultithreadedXRootDSource

        exe_args = {'savemetrics': True,
                    # 'schema': BaseSchema,
                    'schema': NanoAODSchema,
                    'retries': 1}

        out, metrics = processor.run_uproot_job(
            {key: fileset[key] for key in args.samples},
            treename='Events',
            processor_instance=p,
            executor=processor.futures_executor,
            executor_args=exe_args,
            chunksize=10000,
        )

        filehandler = open(f'outfiles/{args.starti}-{args.endi}.pkl', 'wb')
        pickle.dump(out, filehandler)
        filehandler.close()

    elif args.dask:
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
        print(f"Metrics: {metrics}")
        print(f"Finished in {elapsed:.1f}s")

        filehandler = open('out.hist', 'wb')
        pickle.dump(out, filehandler)
        filehandler.close()


if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    parser.add_argument('--outdir',     dest='outdir',     default='outfiles',   help="directory for output files", type=str)
    parser.add_argument("--processor",  dest="processor",  default="trigger",    help="Trigger processor", type=str, choices=['trigger', 'skimmer'])
    parser.add_argument("--dask",       dest="dask",       action="store_true",  default=False, help="Run with dask")
    parser.add_argument("--condor",     dest="condor",     action="store_true",  default=True,  help="Run with condor")
    parser.add_argument('--samples',    dest='samples',    default=[],           help='samples',     nargs='*')
    args = parser.parse_args()

    main(args)
