#!/bin/bash

for txbb_wp in "LP" "MP" "HP"
do 
  for bdt_wp in 0.994 0.99 0.96 0.9 0.8 0.6 0.4
  do 
    cutstr=txbb_${txbb_wp}_bdt_${bdt_wp}
    echo $cutstr
    cd cards/23May13NonresScan/$cutstr

    /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl

    cd -
  done
done