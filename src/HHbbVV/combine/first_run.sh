#!/bin/bash

dataset=data_obs
cards_dir=cards
model_name=test
ws=${model_name}_combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

echo "combine cards"
combineCards.py fail=${cards_dir}/${model_name}/fail.txt failBlinded=${cards_dir}/${model_name}/failBlinded.txt passCat1=${cards_dir}/${model_name}/passCat1.txt passCat1Blinded=${cards_dir}/${model_name}/passCat1Blinded.txt > ${cards_dir}/$ws.txt

echo "text2workspace"
text2workspace.py -D $dataset ${cards_dir}/$ws.txt --channel-masks -o ${cards_dir}/$wsm.root

echo "blinded bkg-only fit snapshot"
combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${cards_dir}/${wsm}.root --verbose 9 --cminDefaultMinimizerStrategy 1 --setParameters mask_passCat1=1,mask_fail=1,mask_passCat1Blinded=0,mask_failBlinded=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin5=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin6=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin7=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin8=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin9=0,r=0 -n Snapshot --freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9,r

echo "asymptotic limit"
combine -M AsymptoticLimits -m 125 -n Cat1 -d ${wsm_snapshot}.root --snapshotName MultiDimFit --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --floatParameters r --toysFrequentist --run blind

echo "fitdiagnostics background-only fit (blinded)"
combine -M FitDiagnostics -d ${wsm_snapshot}.root --setParameters mask_passCat1=1,mask_fail=1,mask_passCat1Blinded=0,mask_failBlinded=0 --saveNormalizations --saveShapes --saveWithUncertainties --saveOverallShapes -n BlindedBkgOnly --ignoreCovWarning --skipSBFit -v 9 --snapshotName MultiDimFit --freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9,r

echo "fitdiagnostics background + signal fit (blinded)"
combine -M FitDiagnostics -d ${wsm_snapshot}.root --setParameters mask_passCat1=1,mask_fail=1,mask_passCat1Blinded=0,mask_failBlinded=0 --saveNormalizations --saveShapes --saveWithUncertainties --saveOverallShapes -n BlindedBkgSig --ignoreCovWarning --skipSBFit -v 9 --snapshotName MultiDimFit --freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9

# echo "fitdiagnostics both signal and background fits (unblinded)"
# combine -M FitDiagnostics -d ${wsm_snapshot}.root --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --rMin -50 --rMax 200 --saveNormalizations --saveShapes --saveWithUncertainties --saveOverallShapes -n Unblinded --ignoreCovWarning --snapshotName MultiDimFit

echo "expected significance"
combine -M Significance -d ${wsm_snapshot}.root --significance -m 125 -n Cat1 --snapshotName MultiDimFit -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1,r=1 --floatParameters r --toysFrequentist
