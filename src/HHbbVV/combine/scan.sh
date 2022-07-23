#!/bin/bash

dataset=data_obs
cutstrs=($(ls -d *))

for cutstr in $cutstrs
do
  echo $cutstr
  cd $cutstr

  ws=${cutstr}_combined
  wsm=${ws}_withmasks

  echo "combining cards"
  combineCards.py fail=fail.txt failBlinded=failBlinded.txt passCat1=passCat1.txt passCat1Blinded=passCat1Blinded.txt > $ws.txt

  echo "text2workspace"
  text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root

  echo "bkg-only fit"
  combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 ${wsm}.root  --verbose 9 --cminDefaultMinimizerStrategy 1 --setParameters mask_fail=1,mask_passCat1=1,r=0 --freezeParameters r

  echo "asymptotic limit"
  combine -M AsymptoticLimits -m 125 -n Cat1 higgsCombineTest.MultiDimFit.mH125.root --snapshotName MultiDimFit --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --floatParameters r --toysFrequentist --run blind | tee asymptoticlimits.txt

  echo "expected significance"
  combine higgsCombineTest.MultiDimFit.mH125.root -M Significance --significance -m 125 -n Cat1 --snapshotName MultiDimFit -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit --setParameters mask_passCat1=0,mask_fail=0,mask_passCat1Blinded=1,mask_failBlinded=1 --floatParameters r --toysFrequentist | tee significance.txt

  cd ..
done
