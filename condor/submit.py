#!/usr/bin/python

import argparse
import os
import re
import fileinput

import json
import glob
import sys

'''
 Submit condor jobs of processor
 Run as e.g.: python submit.py Jul1 run.py 20
 Arguments:
  = [0]: tag of jobs and output dir in eos e.g. Jul1
  - [1]: script to run e.g. run.py (needs to be included in transfer_files in templ.jdl)
  - [2]: number of files per job e.g. 20
'''
# Note: change username in `cmantill` in this script

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('settings', metavar='S', type=str, nargs='+', help='label scriptname (re-tar)')
args = parser.parse_args()

if (not ((len(args.settings) == 3) or (len(args.settings) == 4))):
    print("Wrong number of arguments (must be 3 or 4, found", len(args.settings), ")")
    sys.exit()

label = args.settings[0]
script = args.settings[1]  # should be run.py
files_per_job = int(args.settings[2])

loc_base = os.environ['PWD']
logdir = label
homedir = '/store/user/rkansal/bbVV/' + logdir + '/'
outdir = homedir + '/outfiles/'

# list of samples to run
# TODO
# totfiles = {}
# with open('../data/fileset_2017preUL.json', 'r') as f:
#     newfiles = json.load(f)
#     totfiles.update(newfiles)
# with open('../data/fileset_2017UL.json', 'r') as f:
#     newfiles = json.load(f)
#     totfiles.update(newfiles)
# for sample in samplelist:
#     totfiles[sample] = len(totfiles[sample])

samplelist = {
    'SingleMuon'
}
with open('data/filelist.txt', 'r') as file:
    filelist = [f[:-1] for f in file.readlines()]
fileset = {'2017': filelist}

# name to give your output files
prefix = label

# make local directory
locdir = 'logs/' + logdir
os.system('mkdir -p  %s' %locdir)

# and condor directory
print('CONDOR work dir: ' + outdir)
os.system('mkdir -p /eos/uscms' + outdir)

# submit jobs
nsubmit = 0
for sample in samplelist:

    ## FIXME!
    prefix = sample + '_2017'
    print('Submitting '+ prefix)

    njobs = int(fileset['2017'] / files_per_job) + 1
    remainder = fileset['2017'] - int(files_per_job * (njobs - 1))

    for j in range(njobs):

        condor_templ_file = open(loc_base + "/submit.templ.jdl")
        sh_templ_file     = open(loc_base + "/submit.templ.sh")

        localcondor = locdir + '/' + prefix + "_" + str(j) + ".jdl"
        condor_file = open(localcondor,"w")
        for line in condor_templ_file:
            line=line.replace('DIRECTORY',locdir)
            line=line.replace('PREFIX',prefix)
            line=line.replace('JOBID',str(j))
            condor_file.write(line)
        condor_file.close()

        localsh = locdir + '/' + prefix + "_" + str(j) + ".sh"
        eosoutput = "root://cmseos.fnal.gov/" + outdir + "/" + prefix + '_' + str(j) + '.coffea'
        sh_file = open(localsh, "w")
        for line in sh_templ_file:
            line = line.replace('SCRIPTNAME', script)
            line = line.replace('FILENUM', str(j))
            line = line.replace('YEAR', samplelist[sample])
            line = line.replace('SAMPLE', sample)
            line = line.replace('STARTNUM', str(j * files_per_job))
            line = line.replace('ENDNUM', str((j + 1) * files_per_job))
            line = line.replace('EOSOUT', eosoutput)
            sh_file.write(line)
        sh_file.close()

        os.system('chmod u+x ' + locdir + '/' + prefix + '_' + str(j) + '.sh')
        if (os.path.exists('%s.log' % localcondor)):
            os.system('rm %s.log' % localcondor)
        condor_templ_file.close()
        sh_templ_file.close()

        print('To submit ',localcondor)
        # os.system('condor_submit %s' % localcondor)

        nsubmit = nsubmit + 1

print(nsubmit,"jobs submitted.")
