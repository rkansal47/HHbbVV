#!/bin/bash
# shellcheck disable=SC2154,SC2086,SC2004,SC1091,SC2046,SC1036,SC1088,SC1098

####################################################################################################
# Script for generating toys and doing GoFs for F-tests
# Needs workspaces for all orders + b-only fit snapshot of lowest order model transferred as inputs.
#
# 1) Generates toys using b-only fit
# 2) GoF on toys for lowest and lowest + 1 orders
# 3) Transfers toys and GoF test files to EOS directory
#
# Author: Raghav Kansal
####################################################################################################

echo "Starting job on $$(date)" # Date/time of start of job
echo "Running on: $$(uname -a)" # Condor job is running on this node
echo "System software: $$(cat /etc/redhat-release)" # Operating System on that node

####################################################################################################
# Get my tarred CMSSW with combine already compiled
####################################################################################################

source /cvmfs/cms.cern.ch/cmsset_default.sh
xrdcp -s root://cmseos.fnal.gov//store/user/rkansal/CMSSW_11_3_4.tgz .

echo "extracting tar"
tar -xf CMSSW_11_3_4.tgz
rm CMSSW_11_3_4.tgz
cd CMSSW_11_3_4/src/ || exit
scramv1 b ProjectRename # this handles linking the already compiled code - do NOT recompile
eval $$(scramv1 runtime -sh) # cmsenv is an alias not on the workers
echo "$$CMSSW_BASE is the CMSSW we have on the local worker node"
cd ../..

eosdir="root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/"

echo "Cloning HHbbVV"
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
#move output to eos
xrdcp -f commithash.txt $$eosdir/commithash_${jobnum}.txt
cd .. || exit

outsdir="outs"
scripts_dir="$$(pwd)/HHbbVV/src/HHbbVV/combine"

####################################################################################################
# Generate toys for lowest order polynomial
####################################################################################################

model_name="nTF1_${low1}_nTF2_${low2}"
toys_name="${low1}${low2}"
model_dir="$$(pwd)/$tag/$${model_name}"
echo "Model dir: $${model_dir}"

$${scripts_dir}/run_ftest_res.sh -t --seed $seed --numtoys $num_toys --low1 $low1 --low2 $low2 --cardsdir "." --cardstag $tag --scriptsdir $$scripts_dir --verbose 1

xrdcp $${model_dir}/higgsCombineToys$${toys_name}.GenerateOnly.mH125.$seed.root $$eosdir/$${model_name}/
xrdcp $${model_dir}/$${outsdir}/gentoys$seed.txt $$eosdir/$${model_name}/$${outsdir}/

####################################################################################################
# Run GoF for each order on generated toys
####################################################################################################


$${scripts_dir}/run_ftest_res.sh -f --seed $seed --numtoys $num_toys --low1 $low1 --low2 $low2 --cardsdir "." --cardstag $tag --scriptsdir $$scripts_dir --verbose 1

for (( ord1=$low1; ord1<=$$(($low1 + 1)); ord1++ ))
do
    for (( ord2=$low2; ord2<=$$(($low2 + 1)); ord2++ ))
    do
        if [ $$ord1 -gt $low1 ] && [ $$ord2 -gt $low2 ]
        then
            break
        fi
        model_name="nTF1_$${ord1}_nTF2_$${ord2}"
        cd $tag/$${model_name}/ || exit

        echo "Model: $${model_name}"
        echo $$(pwd)
        ls -lh .

        xrdcp higgsCombineToys$${toys_name}.GoodnessOfFit.mH125.$seed.root $$eosdir/$${model_name}/
        xrdcp $${outsdir}/GoF_toys$${toys_name}$seed.txt $$eosdir/$${model_name}/$${outsdir}/

        rm higgsCombineToys$${toys_name}.GoodnessOfFit.mH125.$seed.root
        rm $${outsdir}/GoF_toys$${toys_name}$seed.txt

        cd - || exit
    done
done
