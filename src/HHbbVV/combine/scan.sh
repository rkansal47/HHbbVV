#!/bin/bash

for txbb_wp in "LP" "MP" "HP"
do 
  # for bdt_wp in 0.6 0.9 0.96 0.99 0.995 0.997 0.998 0.9985
  for bdt_wp in 0.9988
  do 
    cutstr=txbb_${txbb_wp}_bdt_${bdt_wp}
    echo $cutstr

    python3 -u postprocessing/CreateDatacard.py --templates-dir templates/23May13NonresScan/$cutstr \
    --model-name 23May14NonresScan/$cutstr --no-do-jshifts --nTF 0

    cd cards/23May14NonresScan/$cutstr

    /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl

    cd -
  done
done