#!/bin/bash

dataset=data_obs
cards_dir="./"
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir

# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..9}
do
    for channel in fail pass;
    do
        maskunblindedargs+="mask_mXbin${bin}${channel}=1,mask_mXbin${bin}${channel}Blinded=0,"
        maskblindedargs+="mask_mXbin${bin}${channel}=0,mask_mXbin${bin}${channel}Blinded=1,"
    done
done

# remove last comma
maskunblindedargs=${maskunblindedargs%,}
maskblindedargs=${maskblindedargs%,}

# setparams="rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0,CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15=0"
# freezeparams="rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},rgx{.*xhy_mx3000_my190.*},CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15"

setparams="rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0"
freezeparams="rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},rgx{.*xhy_mx3000_my190.*}"

# need to run this for large # of nuisances 
# https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
ulimit -s unlimited

echo "GoF on data"
combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
--snapshotName MultiDimFit --bypassFrequentistFit \
--setParameters ${maskunblindedargs},${setparams},r=0 \
--freezeParameters ${freezeparams},r \
-n passData -v 9 2>&1 | tee $outsdir/GoF_data.txt

echo "GoF on toys"
combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
--snapshotName MultiDimFit --bypassFrequentistFit \
--setParameters ${maskunblindedargs},${setparams},r=0 \
--freezeParameters ${freezeparams},r \
-n passToys -v 9 -s 42 -t 100 --toysFrequentist 2>&1 | tee $outsdir/GoF_toys.txt