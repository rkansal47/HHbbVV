#!/bin/bash

dataset=data_obs
cards_dir=$1
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir


echo "Initial Fit"
combineTool.py -M Impacts -t -1 --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist \
-m 125 -n ".impacts" -d ${wsm_snapshot}.root --doInitialFit --robustFit 1 \
--setParameters mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0 \
--freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9 \
--setParameterRanges r=-20,20 --cminDefaultMinimizerStrategy=1 --saveWorkspace -v 9 2>&1 | tee $outsdir/Impacts_initial.txt


echo "Impacts"
combineTool.py -M Impacts -t -1 --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist \
-m 125 -n ".impacts" -d ${wsm_snapshot}.root --doFits --robustFit 1 \
--setParameters mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0 \
--freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9 \
--setParameterRanges r=-15,15 -v 9 2>&1 | tee $outsdir/Impacts.txt

echo "Collect impacts"
combineTool.py -M Impacts -d ${wsm_snapshot}.root -m 125 -n ".impacts" \
--setParameters mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0 \
--freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9 \
--setParameterRanges r=-20,20 -o impacts.json

echo "Plot impacts"
plotImpacts.py -i impacts.json -o impacts
