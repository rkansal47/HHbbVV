#!/bin/bash
# shellcheck disable=SC2154,SC2046,SC1091,SC2086,SC2155,SC1036,SC1088,SC1098

####################################################################################################
# Template for condor job script to
# 1) Make datacards for specified samples
# 2) Do background-only fit in validation region and get asymptotic limits
#
# Author: Raghav Kansal
####################################################################################################

echo "Starting job on $$(date)" # Date/time of start of job
echo "Running on: $$(uname -a)" # Condor job is running on this node
echo "System software: $$(cat /etc/redhat-release)" # Operating System on that node

####################################################################################################
# Get my tarred CMSSW with combine already compiled
# Made with `tar cvfz CMSSW_11_3_4.tgz CMSSW_11_3_4`
####################################################################################################

source /cvmfs/cms.cern.ch/cmsset_default.sh
xrdcp -s root://cmseos.fnal.gov//store/user/rkansal/CMSSW_11_3_4.tgz .

echo "extracting tar"
tar -xf CMSSW_11_3_4.tgz
rm CMSSW_11_3_4.tgz
cd CMSSW_11_3_4/src/ || exit
scramv1 b ProjectRename # this handles linking the already compiled code - do NOT recompile
eval $$(scramv1 runtime -sh) # cmsenv is an alias not on the workers
echo $$CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ../.. || exit

echo "testing combine"
combine

####################################################################################################
# Install Python Packages
# Need to install with --user, after changing the user directory using the PYTHONUSERBASE arg
# (Can't install with normal --user since condor job doesn't have write access to my user dir)
# See https://stackoverflow.com/a/29103053/3759946
####################################################################################################

mkdir local_python
export PYTHONUSERBASE=$(pwd)/local_python

echo "Installing hist"
pip3 install --user hist==2.7.2
git clone -b square_coef https://github.com/rkansal47/rhalphalib.git
cd rhalphalib || exit
echo "Installing rhalphalib"
pip3 install --user .
cd .. || exit

export PYTHONPATH=$(pwd)/local_python/lib/python3.8/site-packages/:$PYTHONPATH

echo "testing installed libraries"
echo "import hist; import rhalphalib; print('Import successful! hist version:', hist.__version__)" > lib_test.py
python3 lib_test.py

ls -lh .

####################################################################################################
# Get templates from EOS directory and run fit script
####################################################################################################

mkdir templates
mkdir cards

templates_dir=${in_templates_dir}
cards_dir=${in_cards_dir}

# get backgrounds templates
xrdcp -r root://cmseos.fnal.gov/${templates_dir}/backgrounds templates/

for sample in $samples
do
    echo -e "\n\n$sample"

    # get sample templates
    xrdcp -r root://cmseos.fnal.gov/${templates_dir}/$sample templates/

    python3 -u postprocessing/CreateDatacard.py --templates-dir templates --sig-separate --resonant \
    --model-name $sample --sig-sample $sample ${datacard_args}

    cd cards/$sample || exit
    ../../combine/run_blinded.sh --workspace --bfit --limits --resonant
    cd ../.. || exit

    # transfer output cards
    xrdcp -r cards/$sample root://cmseos.fnal.gov/${cards_dir}/$sample/
    rm -rf templates/$sample
done

rm -rf templates/backgrounds
