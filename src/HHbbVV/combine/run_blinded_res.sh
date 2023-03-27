#!/bin/bash

dataset=data_obs
cards_dir=$1
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir


echo "combine cards"

# one channel per bin
ccargs=""
for bin in {0..23}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=${cards_dir}/mXbin${bin}${channel}.txt "
    done
done

combineCards.py $ccargs > $ws.txt

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