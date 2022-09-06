#!/bin/bash

dataset=data_obs
cards_dir=$1
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir


echo "combine cards"
combineCards.py fail=${cards_dir}/fail.txt failBlinded=${cards_dir}/failBlinded.txt passCat1=${cards_dir}/passCat1.txt passCat1Blinded=${cards_dir}/passCat1Blinded.txt > $ws.txt

echo "text2workspace"
text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root > $outsdir/text2workspace.txt

# echo "blinded bkg-only fit snapshot"
# combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${cards_dir}/${wsm}.root --verbose 9 --cminDefaultMinimizerStrategy 1 --setParameters mask_passCat1=1,mask_fail=1,mask_passCat1Blinded=0,mask_failBlinded=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin5=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin6=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin7=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin8=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin9=0,r=0 -n Snapshot --freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9,r > $outsdir/MultiDimFit.txt

# echo "asymptotic limit"
# combine -M AsymptoticLimits -m 125 -n Cat1 -d ${wsm_snapshot}.root --snapshotName MultiDimFit --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --floatParameters r --toysFrequentist --run blind > $outsdir/AsymptoticLimits.txt

# echo "expected significance"
# combine -M Significance -d ${wsm_snapshot}.root --significance -m 125 -n Cat1 --snapshotName MultiDimFit -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1,r=1 --floatParameters r --toysFrequentist > $outsdir/Significance.txt

echo "fitdiagnostics"
combine -M FitDiagnostics -m 125 -d ${wsm}.root --setParameters mask_passCat1=1,mask_fail=1,mask_passCat1Blinded=0,mask_failBlinded=0 --saveNormalizations --saveShapes --saveWithUncertainties --saveOverallShapes -n Blinded --ignoreCovWarning -v 9 --freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9

