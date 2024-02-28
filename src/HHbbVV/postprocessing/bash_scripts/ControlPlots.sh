#!/bin/bash

####################################################################################################
# Script for Control Plots
# Author: Raghav Kansal
####################################################################################################


####################################################################################################
# Read options
####################################################################################################

MAIN_DIR="../../.."
TAG=24Jan13Res
resonant="--resonant"
samples="HHbbVV VBFHHbbVV NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80 NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150"
hem2d="--HEM2d"

options=$(getopt -o "" --long "nonresonant,nohem2d,tag:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --nonresonant)
            resonant=""
            samples="HHbbVV qqHH_CV_1_C2V_1_kl_1_HHbbVV"
            ;;
        --nohem2d)
            hem2d=""
            ;;
        --tag)
            shift
            TAG=$1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

for year in 2016APV 2016 2017 2018
do
    python -u postprocessing.py --control-plots --year $year ${resonant} ${hem2d} \
    --data-dir "${MAIN_DIR}/../data/skimmer/Feb24" \
    --signal-data-dirs "${MAIN_DIR}/../data/skimmer/Jun10" "${MAIN_DIR}/../data/skimmer/Apr11" \
    --sig-samples ${samples} \
    --bdt-preds-dir "${MAIN_DIR}/../data/skimmer/Feb24/23_05_12_multiclass_rem_feats_3/inferences" \
    --plot-dir "${MAIN_DIR}/plots/PostProcessing/$TAG"
done