#!/bin/bash

dataset=data_obs
cards_dir=$1
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir


# one channel per bin
ccargs=""
for bin in {0..18}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=${cards_dir}/mXbin${bin}${channel}.txt "
    done
done

# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..18}
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

setparams="rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0,CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15=0"
freezeparams="rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},rgx{.*xhy_mx3000_my190.*},CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15"

setparamsunblinded="rgx{passBlinded_.*mcstat.*}=0,rgx{failBlinded_.*mcstat.*}=0,CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15=0"
freezeparamsunblinded="rgx{passBlinded_.*mcstat.*},rgx{failBlinded_.*mcstat.*},rgx{.*xhy_mx3000_my190.*},CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15"

echo "mask unblinded args:"
echo $maskunblindedargs

# need to run this for large # of nuisances 
# https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
ulimit -s unlimited

# echo "combine cards"
# combineCards.py $ccargs > $ws.txt

# echo "text2workspace"
# text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt

# echo "blinded bkg-only fit snapshot"
# combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v 9 --cminDefaultMinimizerStrategy 1 \
# --setParameters ${maskunblindedargs},${setparams},r=0  \
# --freezeParameters r,${freezeparams} \
# -n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt

# echo "asymptotic limit"
# combine -M AsymptoticLimits -m 125 -n pass -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v 9 \
# --saveWorkspace --saveToys --bypassFrequentistFit \
# --setParameters ${maskblindedargs},${setparamsunblinded} \
# --freezeParameters ${freezeparamsunblinded} \
# --floatParameters r --toysFrequentist --run blind 2>&1 | tee $outsdir/AsymptoticLimits.txt

echo "expected significance"
combine -M Significance -d ${wsm_snapshot}.root --significance -m 125 -n pass --snapshotName MultiDimFit -v 9 \
-t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit \
--setParameters ${maskblindedargs},${setparamsunblinded},r=1 \
--freezeParameters ${freezeparamsunblinded} \
--floatParameters r --toysFrequentist 2>&1 | tee $outsdir/Significance.txt

# echo "fitdiagnostics"
# combine -M FitDiagnostics -m 125 -d ${wsm}.root \
# --setParameters ${maskunblindedargs},${setparams},r=0 \
# --freezeParameters rgx{pass_.*mcstat.*},${freezeparams},r \
# --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes \
# -n Blinded --ignoreCovWarning -v 9 2>&1 | tee $outsdir/FitDiagnostics.txt