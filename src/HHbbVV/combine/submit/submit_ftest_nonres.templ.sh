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

# inputs

tag=${in_tag}
low1=${in_low1}
seed=${in_seed}
num_toys=${in_num_toys}

cards_dir=$tag

####################################################################################################
# Fit args
####################################################################################################

wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="outs"

CMS_PARAMS_LABEL="CMS_bbWW_hadronic"

# nonresonant args
ccargs="fail=${cards_dir}/fail.txt failBlinded=${cards_dir}/failBlinded.txt pass=${cards_dir}/pass.txt passBlinded=${cards_dir}/passBlinded.txt"
maskunblindedargs="mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0"
maskblindedargs="mask_pass=0,mask_fail=0,mask_passBlinded=1,mask_failBlinded=1"

# freeze qcd params in blinded bins
setparamsblinded=""
freezeparamsblinded=""
for bin in {5..9}
do
    setparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin}=0,"
    freezeparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin},"
done

# remove last comma
setparamsblinded=${setparamsblinded%,}
freezeparamsblinded=${freezeparamsblinded%,}


# floating parameters using var{} floats a bunch of parameters which shouldn't be floated,
# so countering this inside --freezeParameters which takes priority.
# Although, practically even if those are set to "float", I didn't see them ever being fitted,
# so this is just to be extra safe.
unblindedparams="--freezeParameters var{.*_In},var{.*__norm},var{n_exp_.*} --setParameters $maskblindedargs"

####################################################################################################
# Generate toys for lowest order polynomial
####################################################################################################

model_name="nTF_${low1}"
toys_name="${low1}"
cd ${cards_dir}/${model_name}/ || exit
mkdir -p $outsdir

ulimit -s unlimited

echo "Toys for $low1 order fit"
combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
--snapshotName MultiDimFit --bypassFrequentistFit \
--setParameters ${maskunblindedargs},${setparams},r=0 \
--freezeParameters ${freezeparams},r \
-n "Toys${toys_name}" -t $num_toys --saveToys -s $seed -v 9 2>&1 | tee $outsdir/gentoys$seed.txt

toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
xrdcp $toys_file root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/
xrdcp $outsdir/gentoys$seed.txt root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/$outsdir/

cd - || exit

####################################################################################################
# Run GoF for each order on generated toys
####################################################################################################

for (( ord1=$low1; ord1<=$((low1 + 1)); ord1++ ))
do
    model_name="nTF_${ord1}"
    echo "GoF for $model_name"

    cd "${cards_dir}/${model_name}/" || exit
    mkdir -p $outsdir

    ulimit -s unlimited

    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    --setParameters "${maskunblindedargs},${setparams},r=0" \
    --freezeParameters "${freezeparams},r" \
    -n "Toys${toys_name}Seed$seed" -v 9 -s $seed -t $num_toys \
    --toysFile ${toys_file} 2>&1 | tee $outsdir/GoF_toys${toys_name}$seed.txt

    xrdcp "higgsCombineToys${toys_name}Seed$seed.GoodnessOfFit.mH125.$seed.root" "root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/"
    xrdcp "$outsdir/GoF_toys${toys_name}$seed.txt" "root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/$outsdir/"

    rm "higgsCombineToys${toys_name}Seed$seed.GoodnessOfFit.mH125.$seed.root"
    rm "$outsdir/GoF_toys${toys_name}$seed.txt"

    cd - || exit
done
