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
export PYTHONUSERBASE=$$(pwd)/local_python

echo "Installing hist"
pip3 install --user hist==2.7.2

echo "Installing rhalphalib"
# try 3 times in case of network errors
(
    r=3
    # shallow clone of single branch (keep repo size as small as possible)
    while ! git clone https://github.com/rkansal47/rhalphalib.git
    do
        ((--r)) || exit
        sleep 60
    done
)
cd rhalphalib || exit
pip3 install --user .
cd .. || exit

echo "Installing HHbbVV"
# try 3 times in case of network errors
(
    r=3
    # shallow clone of single branch (keep repo size as small as possible)
    while ! git clone --single-branch --branch $branch --depth=1 https://github.com/$gituser/HHbbVV/
    do
        ((--r)) || exit
        sleep 60
    done
)
cd HHbbVV || exit

commithash=$$(git rev-parse HEAD)
echo "https://github.com/rkansal47/HHbbVV/commit/$${commithash}" > commithash.txt

#move output to t2s
xrdcp -f commithash.txt root://cmseos.fnal.gov/${cards_dir}/commithash_${jobnum}.txt

pip3 install --user .

export PYTHONPATH=$$(pwd)/local_python/lib/python3.8/site-packages/:$$PYTHONPATH

echo "testing installed libraries"
echo "import hist; import rhalphalib; import HHbbVV; print('Import successful! hist version:', hist.__version__)" > lib_test.py
python3 lib_test.py

ls -lh .

####################################################################################################
# Get templates from EOS directory and run fit script
####################################################################################################

cd src/HHbbVV/ || exit
mkdir -p templates/backgrounds
mkdir -p cards
qcd_fit_dir=$$(pwd)/combine/qcdfit22sl7

# get backgrounds templates
for file in "2016_templates.pkl" "2016APV_templates.pkl" "2017_templates.pkl" "2018_templates.pkl" "systematics.json"
do
    xrdcp -r root://redirector.t2.ucsd.edu:1095//${templates_dir}/backgrounds/$${file} templates/backgrounds/
done

for sample in $samples
do
    echo -e "\n\n$${sample}"
    mkdir -p templates/$${sample}

    # get sample templates
    for file in "2016_templates.pkl" "2016APV_templates.pkl" "2017_templates.pkl" "2018_templates.pkl" "systematics.json"
    do
        xrdcp -r root://redirector.t2.ucsd.edu:1095//${templates_dir}/$${sample}/$${file} templates/$${sample}/
    done

    python3 -u postprocessing/CreateDatacard.py --templates-dir "templates/$${sample}" --bg-templates-dir "templates/backgrounds" \
    --sig-separate --resonant --model-name $${sample} --sig-sample $${sample} ${datacard_args} --qcd-fit-dir $qcd_fit_dir

    cd cards/$${sample} || exit
    ../../combine/$script --workspace --bfit --limits --resonant --significance
    cd ../.. || exit

    echo -e "\n\n\n"
    echo "Finished $${sample}"
    echo "Outputs:"
    ls -lh cards/$${sample}

    # transfer output cards
    xrdcp -r cards/$${sample} root://cmseos.fnal.gov/${cards_dir}/$${sample}
    rm -rf templates/$${sample}
    rm -rf cards/$${sample}
done

rm -rf templates/backgrounds
