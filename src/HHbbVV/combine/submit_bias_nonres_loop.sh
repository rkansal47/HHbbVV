#!/bin/bash
# shellcheck disable=SC2043

#######################################################################
# Script to submit bias jobs, needs to be run from inside the datacards directory
#####################################################

seed=$1
TAG=$2

for bias in 0.0 0.3 1.0 1.5
do
  python3 /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/submit/submit_bias.py --seed "$seed" --num-jobs 100 --toys-per-job 10 --bias "$bias" --submit --tag "$TAG"
done

# # need to submit extra jobs for these because of high fit failures
# sample=NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250
# cd $sample
# bias=0.0
# python3 /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/submit/submit_bias.py --seed $((seed + 1000)) --num-jobs 100 --toys-per-job 10 --bias $bias --submit --tag $TAG

# bias=0.15
# python3 /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/submit/submit_bias.py --seed $((seed + 1000)) --num-jobs 50 --toys-per-job 10 --bias $bias --submit --tag $TAG
# cd -
