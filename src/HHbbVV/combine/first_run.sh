#!/bin/bash

dataset=data_obs
cards_dir=cards
model_name=test
ws=${model_name}_combined
wsm=${ws}_withmasks

combineCards.py fail=${cards_dir}/${model_name}/fail.txt failBlinded=${cards_dir}/${model_name}/failBlinded.txt passCat1=${cards_dir}/${model_name}/passCat1.txt passCat1Blinded=${cards_dir}/${model_name}/passCat1Blinded.txt > ${cards_dir}/$ws.txt

text2workspace.py -D $dataset ${cards_dir}/$ws.txt --channel-masks -o ${cards_dir}/$wsm.root

echo "bkg-only fit"
combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 ${cards_dir}/${wsm}.root  --verbose 9 --cminDefaultMinimizerStrategy 1 --setParameters mask_fail=1,mask_passCat1=1,r=0 --freezeParameters r

echo "asymptotic limit"
combine -M AsymptoticLimits -m 125 -n Cat1 higgsCombineTest.MultiDimFit.mH125.root --snapshotName MultiDimFit --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --floatParameters r --toysFrequentist --run blind

echo "FitDiagnostic S=0"
combine -M FitDiagnostics ${cards_dir}/${wsm}.root --setParameters mask_fail=1,mask_passCat1=1 --rMin 0 --rMax 2 --saveNormalizations --saveShapes --saveWithUncertainties --saveOverallShapes -n SBplusfail --ignoreCovWarning

echo "FitDiagnostic S=1"
combine -M FitDiagnostics ${cards_dir}/${wsm}.root --setParameters mask_fail=1,mask_passCat1=1 --rMin 1 --rMax 1 --skipBOnlyFit --saveNormalizations --saveShapes --saveWithUncertainties --saveOverallShapes -n SBplusfailSfit --ignoreCovWarning

echo "expected significance"
combine higgsCombineTest.MultiDimFit.mH125.root -M Significance --significance -m 125 -n Cat1 --snapshotName MultiDimFit -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --floatParameters r --toysFrequentist
