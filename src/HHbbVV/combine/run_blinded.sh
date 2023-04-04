#!/bin/bash

dataset=data_obs
cards_dir="./"
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir

maskunblindedargs="mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0"
maskblindedargs="mask_pass=0,mask_fail=0,mask_passBlinded=1,mask_failBlinded=1"

# freeze qcd params in blinded bins
setqcdparams=""
freezeqcdparams=""
for bin in {5..9}
do
    setqcdparams+="CMS_bbWW_boosted_ggf_qcdparam_msdbin${bin}=0,"
    freezeqcdparams+="CMS_bbWW_boosted_ggf_qcdparam_msdbin${bin},"
done

# remove last comma
setqcdparams=${setqcdparams%,}
freezeqcdparams=${freezeqcdparams%,}

echo "combine cards"
combineCards.py fail=${cards_dir}/fail.txt failBlinded=${cards_dir}/failBlinded.txt pass=${cards_dir}/pass.txt passBlinded=${cards_dir}/passBlinded.txt > $ws.txt

echo "text2workspace"
text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt

echo "blinded bkg-only fit snapshot"
combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root --verbose 9 --cminDefaultMinimizerStrategy 1 \
--setParameters ${maskunblindedargs},${setqcdparams},r=0 \
--freezeParameters ${freezeqcdparams},r \
-n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt

echo "asymptotic limit"
combine -M AsymptoticLimits -m 125 -n pass -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v 9 \
--saveWorkspace --saveToys --bypassFrequentistFit \
--setParameters ${maskblindedargs} \
--floatParameters r --toysFrequentist --run blind 2>&1 | tee $outsdir/AsymptoticLimits.txt

echo "expected significance"
combine -M Significance -d ${wsm_snapshot}.root --significance -m 125 -n pass --snapshotName MultiDimFit -v 9 \
-t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit \
--setParameters ${maskblindedargs},r=1 \
--floatParameters r --toysFrequentist 2>&1 | tee $outsdir/Significance.txt

echo "fitdiagnostics"
combine -M FitDiagnostics -m 125 -d ${wsm}.root \
--setParameters ${maskunblindedargs},${setqcdparams} \
--freezeParameters ${freezeqcdparams} \
--saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes -n Blinded \
--ignoreCovWarning -v 9 2>&1 | tee $outsdir/FitDiagnostics.txt

