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
for bin in {0..18}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=${cards_dir}/mXbin${bin}${channel}.txt "
        maskunblindedargs+=""
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

combineCards.py $ccargs > $ws.txt

echo "text2workspace"
text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 
# 2>&1 | tee $outsdir/text2workspace.txt

# echo "blinded bkg-only fit snapshot"
# combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v 9 --cminDefaultMinimizerStrategy 1 \
# --setParameters ${maskunblindedargs}r=0,rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0,CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15=0  \
# --freezeParameters r,rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15 \
# -n Snapshot 
# 2>&1 | tee $outsdir/MultiDimFit.txt

# echo "fitdiagnostics"
# combine -M FitDiagnostics -m 125 -d ${wsm}.root \
# --setParameters ${maskunblindedargs},rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0,CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15=0,r=0 \
# --freezeParameters rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},CMS_XHYbbWW_boosted_qcdparam_mXbin0_mYbin15,r \
# --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes \
# -n Blinded --justFit --ignoreCovWarning -v 9 2>&1 | tee $outsdir/FitDiagnostics.txt