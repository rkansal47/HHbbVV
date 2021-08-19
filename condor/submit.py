#!/usr/bin/python

import argparse
import os
from math import ceil


def get_fileset(ptype):
    print(ptype)
    if ptype == 'trigger':
        with open('data/SingleMuon_2017.txt', 'r') as file:
            filelist = [f[:-1] for f in file.readlines()]

        files = {'2017': filelist}
        fileset = {k: files[k] for k in files.keys()}
        return fileset

    elif ptype == 'skimmer':
        from os import listdir

        # TODO: replace with UL sample once we have it
        with open('data/2017_preUL_nano/HHToBBVVToBBQQQQ_cHHH1.txt', 'r') as file:
            filelist = [f[:-1].replace('/eos/uscms/', 'root://cmsxrootd.fnal.gov//') for f in file.readlines()]   # need to use xcache redirector at Nebraksa coffea-casa

        fileset = {
            '2017_HHToBBVVToBBQQQQ_cHHH1': filelist
        }

        # extra samples in the folder we don't need for this analysis - TODO: should instead have a list of all samples we need
        ignore_samples = ['GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8',
                          'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
                          'ST_tW_antitop_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8',
                          'ST_tW_top_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8']

        for sample in listdir('data/2017_UL_nano/'):
            if sample[-4:] == '.txt' and sample[:-4] not in ignore_samples:
                with open(f'data/2017_UL_nano/{sample}', 'r') as file:
                    if 'JetHT' in sample: filelist = [f[:-1].replace('/hadoop/cms/', 'root://redirector.t2.ucsd.edu//') for f in file.readlines()]
                    else: filelist = [f[:-1].replace('/eos/uscms/', 'root://cmsxrootd.fnal.gov//') for f in file.readlines()]

                fileset['2017_' + sample[:-4].split('_TuneCP5')[0]] = filelist

        return fileset


def main(args):
    locdir = 'condor/' + args.tag
    homedir = f'/store/user/rkansal/bbVV/{args.processor}/'
    outdir = homedir + args.tag + '/outfiles/'

    # make local directory
    logdir = locdir + '/logs'
    os.system(f'mkdir -p {logdir}')

    # and condor directory
    print('CONDOR work dir: ' + outdir)
    os.system(f'mkdir -p /eos/uscms/{outdir}')

    fileset = get_fileset(args.processor)

    # directories for every sample
    for sample in fileset:
        os.system(f'mkdir -p /eos/uscms/{outdir}/{sample}')

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        print('Submitting ' + sample)

        tot_files = len(fileset[sample])
        njobs = ceil(tot_files / args.files_per_job)

        for j in range(njobs):
            if j == 3: break
            condor_templ_file = open("condor/submit.templ.jdl")

            localcondor = f'{locdir}/{sample}_{j}.jdl'
            condor_file = open(localcondor, "w")
            for line in condor_templ_file:
                line = line.replace('DIRECTORY', locdir)
                line = line.replace('PREFIX', sample)
                line = line.replace('JOBID', str(j))
                condor_file.write(line)

            condor_file.close()
            condor_templ_file.close()


            sh_templ_file = open("condor/submit.templ.sh")

            localsh = f'{locdir}/{sample}_{j}.sh'
            eosoutput = f'root://cmseos.fnal.gov/{outdir}/{sample}/out_{j}.pkl'
            sh_file = open(localsh, "w")
            for line in sh_templ_file:
                line = line.replace('SCRIPTNAME', args.script)
                line = line.replace('YEAR', args.year)
                line = line.replace('SAMPLE', sample)
                line = line.replace('PROCESSOR', args.processor)
                line = line.replace('STARTNUM', str(j * args.files_per_job))
                line = line.replace('ENDNUM', str((j + 1) * args.files_per_job))
                line = line.replace('EOSOUT', eosoutput)
                sh_file.write(line)
            sh_file.close()
            sh_templ_file.close()

            os.system(f'chmod u+x {localsh}')
            if (os.path.exists('%s.log' % localcondor)):
                os.system('rm %s.log' % localcondor)

            print('To submit ', localcondor)
            # os.system('condor_submit %s' % localcondor)

            nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--script',       default='run.py',       help="script to run",                       type=str)
    parser.add_argument('--year',       dest='year',       default='2017',       help="year",                       type=str)
    parser.add_argument('--tag',        dest='tag',        default='Test',       help="process tag",                type=str)
    parser.add_argument('--outdir',     dest='outdir',     default='outfiles',   help="directory for output files", type=str)
    parser.add_argument("--processor",  dest="processor",  default="trigger",    help="which processor",          type=str, choices=['trigger', 'skimmer'])
    parser.add_argument("--files-per-job", default=20,    help="# files per condor job",          type=int)
    args = parser.parse_args()

    main(args)
