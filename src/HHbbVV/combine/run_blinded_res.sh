#!/bin/bash

dataset=data_obs
cards_dir=$1
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir


echo "combine cards"
combineCards.py fail=${cards_dir}/fail.txt failBlinded=${cards_dir}/failBlinded.txt pass=${cards_dir}/pass.txt passBlinded=${cards_dir}/passBlinded.txt > $ws.txt

echo "text2workspace"
text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt

echo "blinded bkg-only fit snapshot"
combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root --verbose 9 --cminDefaultMinimizerStrategy 1 \
--setParameters mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0,r=0 \
--freezeParameters r \
-n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt

echo "fitdiagnostics"
combine -M FitDiagnostics -m 125 -d ${wsm}.root \
--setParameters mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0 \
--saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes -n Blinded --ignoreCovWarning -v 9 2>&1 | tee $outsdir/FitDiagnostics.txt

