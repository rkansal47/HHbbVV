#!/bin/bash                                                                                                                   
echo "Starting job on " `date` #Date/time of start of job                                                                       
echo "Running on: `uname -a`" #Condor job is running on this node                                                               
echo "System software: `cat /etc/redhat-release`" #Operating System on that node                                             

# CMSSW
source /cvmfs/cms.cern.ch/cmsset_default.sh 
xrdcp -s root://cmseos.fnal.gov//store/user/rkansal/CMSSW_11_2_0.tgz .

echo "extracting tar"
tar -xf CMSSW_11_2_0.tgz
rm CMSSW_11_2_0.tgz
cd CMSSW_11_2_0/src/
scramv1 b ProjectRename # this handles linking the already compiled code - do NOT recompile
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ../..

# Combine                                                                                                                   
# git clone -b py3 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
# # git clone -b v2.0.0 https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
# echo "Building combine"
# scramv1 b clean; scramv1 b

echo "testing combine"
combine


echo "working dir"
pwd

# echo "installing user"
# pip3 install --user hist

# python packages
mkdir local_python
export PYTHONUSERBASE=$(pwd)/local_python

echo "python user base"
echo $PYTHONUSERBASE

echo "Installing hist"
pip3 install --user hist
git clone -b square_coef https://github.com/rkansal47/rhalphalib.git
cd rhalphalib
echo "Installing rhalphalib"
pip3 install --user .
cd ..

echo "working dir"
pwd
ls -lh .

echo "local python"
ls -lh local_python
ls -lh local_python/lib
ls -lh local_python/lib/python3.8/site-packages/

echo "full python path"
ls -lh $pwd/local_python/lib/python3.8/site-packages/

export PYTHONPATH=$(pwd)/local_python/lib/python3.8/site-packages/:$PYTHONPATH

echo "testing hist"
echo "import hist" > hist_test.py
cat hist_test.py
python3 hist_test.py
echo "hist test over"

echo "testing rh"
echo "import rhalphalib" > rh_test.py
cat rh_test.py
python3 rh_test.py
echo "rh test over"

echo "python path"
echo $PYTHONPATH

mkdir templates
mkdir cards

tag=Apr11

xrdcp -r root://cmseos.fnal.gov//store/user/rkansal/bbVV/templates/Apr11/backgrounds templates/

for sample in $samples
do
    echo -e "\n\n$sample"
    xrdcp -r root://cmseos.fnal.gov//store/user/rkansal/bbVV/templates/Apr11/$sample templates/

    python3 postprocessing/CreateDatacard.py --templates-dir templates --sig-separate --resonant \
    --model-name $sample --sig-sample $sample

    cd cards/$sample
    ../../combine/run_blinded_res.sh
    cd ../..

    xrdcp -r cards/$sample root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/$tag/$sample/
    rm -rf templates/$sample
done

rm -rf templates/backgrounds