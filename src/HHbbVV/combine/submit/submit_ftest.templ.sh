#!/bin/bash 

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

echo "Starting job on " `date` #Date/time of start of job                                                                       
echo "Running on: `uname -a`" #Condor job is running on this node                                                               
echo "System software: `cat /etc/redhat-release`" #Operating System on that node                                             

####################################################################################################
# Get my tarred CMSSW with combine already compiled
####################################################################################################

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

# inputs

tag=${in_tag}
low1=${in_low1}
low2=${in_low2}
seed=${in_seed}
num_toys=${in_num_toys}

cards_dir=$tag

####################################################################################################
# Fit args
####################################################################################################

dataset=data_obs
ws="./combined"
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="outs"

# one channel per bin
ccargs=""
# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..9}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=mXbin${bin}${channel}.txt "
    done

    for channel in fail pass;
    do
        maskunblindedargs+="mask_mXbin${bin}${channel}=1,mask_mXbin${bin}${channel}Blinded=0,"
        maskblindedargs+="mask_mXbin${bin}${channel}=0,mask_mXbin${bin}${channel}Blinded=1,"
    done
done

# remove last comma
maskunblindedargs=${maskunblindedargs%,}
maskblindedargs=${maskblindedargs%,}

setparams="rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0"
freezeparams="rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},rgx{.*xhy_mx.*}"

####################################################################################################
# Generate toys for lowest order polynomial
####################################################################################################

model_name="nTF1_${low1}_nTF2_${low2}"
toys_name="${low1}${low2}"
cd ${cards_dir}/${model_name}/
mkdir -p $outsdir

ulimit -s unlimited

echo "Toys for ($low1, $low2) order fit"
combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
--snapshotName MultiDimFit --bypassFrequentistFit \
--setParameters ${maskunblindedargs},${setparams},r=0 \
--freezeParameters ${freezeparams},r \
-n "Toys${toys_name}" -t $num_toys --saveToys -s $seed -v 9 2>&1 | tee $outsdir/gentoys$seed.txt

toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
xrdcp $toys_file root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/
xrdcp $outsdir/gentoys$seed.txt root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/$outsdir/

cd -

####################################################################################################
# Run GoF for each order on generated toys
####################################################################################################

for (( ord1=$low1; ord1<=$((low1 + 1)); ord1++ ))
do
    for (( ord2=$low2; ord2<=$((low2 + 1)); ord2++ ))
    do
        if [ $ord1 -gt $low1 ] && [ $ord2 -gt $low2 ]
        then
            break
        fi

        model_name="nTF1_${ord1}_nTF2_${ord2}"
        echo "GoF for $model_name"

        cd ${cards_dir}/${model_name}/
        mkdir -p $outsdir

        ulimit -s unlimited

        combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
        --setParameters ${maskunblindedargs},${setparams},r=0 \
        --freezeParameters ${freezeparams},r \
        -n Toys${toys_name}Seed$seed -v 9 -s $seed -t $num_toys \
        --toysFile ${toys_file} 2>&1 | tee $outsdir/GoF_toys${toys_name}$seed.txt

        xrdcp "higgsCombineToys${toys_name}Seed$seed.GoodnessOfFit.mH125.$seed.root" root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/
        xrdcp $outsdir/GoF_toys${toys_name}$seed.txt root://cmseos.fnal.gov//store/user/rkansal/bbVV/cards/f_tests/$tag/$model_name/$outsdir/

        rm "higgsCombineToys${toys_name}Seed$seed.GoodnessOfFit.mH125.$seed.root" 
        rm "$outsdir/GoF_toys${toys_name}$seed.txt"

        cd -
    done
done