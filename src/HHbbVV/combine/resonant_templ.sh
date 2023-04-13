#!/bin/bash                                                                                                                   
echo "Starting job on " `date` #Date/time of start of job                                                                       
echo "Running on: `uname -a`" #Condor job is running on this node                                                               
echo "System software: `cat /etc/redhat-release`" #Operating System on that node                                             

# CMSSW                                                                                                                        
source /cvmfs/cms.cern.ch/cmsset_default.sh
scramv1 project CMSSW CMSSW_11_2_0 # cmsrel is an alias not on the workers                                                        
ls -alrth
cd CMSSW_11_2_0/src/
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers                                                          
echo $CMSSW_BASE "is the CMSSW we created on the local worker node"

# Combine                                                                                                                   
git clone -b py3 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
# git clone -b v2.0.0 https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
echo "Building combine"
scramv1 b clean; scramv1 b

# python packages
echo "Installing hist"
pip3 install --user hist
git clone -b square_coef https://github.com/rkansal47/rhalphalib.git
cd rhalphalib
echo "Installing rhalphalib"
pip3 install --user .

cd ..
mkdir templates
mkdir cards

tag=Apr11

for sample in $samples
do
    echo -e "\n\n$sample"
    xrdcp -r root://cmseos.fnal.gov//store/user/rkansal/bbVV/templates/$tag/$sample templates/

    python3 postprocessing/CreateDatacard.py --templates-dir templates --sig-separate --resonant \
    --model-name $sample --sig-sample $sample

    cd cards/$sample
    ../../combine/run_blinded_res.sh
    cd ../..

    xrdcp -r cards/$sample root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/$tag/
    rm -rf templates/$sample
done