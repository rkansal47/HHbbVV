#!/bin/bash

####################################################################################################
# Script for running bias test
#
# Author: Raghav Kansal
####################################################################################################

echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

####################################################################################################
# Get my tarred CMSSW with combine already compiled
####################################################################################################

source /cvmfs/cms.cern.ch/cmsset_default.sh
xrdcp -s root://cmseos.fnal.gov//store/user/rkansal/CMSSW_11_3_4.tgz .

echo "extracting tar"
tar -xf CMSSW_11_3_4.tgz
rm CMSSW_11_3_4.tgz
cd CMSSW_11_3_4/src/
scramv1 b ProjectRename # this handles linking the already compiled code - do NOT recompile
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ../..

ls -lh

ulimit -s unlimited
$command
