#!/bin/bash
# shellcheck disable=SC2086

seed_last_digit=$1
TAG=$2

for sample in NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250
do
  cd $sample || exit
  mkdir -p bias/$TAG
  # move only files with size > 0
  find higgsCombinebias*.FitDiagnostics.mH125.*$seed_last_digit.root -type f -size +0c -exec mv {} bias/$TAG/ \;
  # mv higgsCombinebias*.FitDiagnostics.mH125.*$seed_last_digit.root bias/$TAG/
  cd - || exit
done
