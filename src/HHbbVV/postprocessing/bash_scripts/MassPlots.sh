#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Script for AK8 Jet Msd and Regressed mass plots without filters, so we can see the peak at 0 in Msd
# Author: Raghav Kansal
####################################################################################################


####################################################################################################
# Options
# --tag: Tag for the plots
####################################################################################################

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
TAG=""
samples="HHbbVV VBFHHbbVV NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80 NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150"
resonant="--resonant"

options=$(getopt -o "" --long "nonresonant,tag:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --nonresonant)
            resonant=""
            samples="HHbbVV VBFHHbbVV qqHH_CV_1_C2V_0_kl_1_HHbbVV qqHH_CV_1_C2V_2_kl_1_HHbbVV"
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

if [[ -z $TAG ]]; then
  echo "Tag required using the --tag option. Exiting"
  exit 1
fi

# for year in 2016APV 2016 2017 2018
for year in 2016APV 2016 2017
do
    python -u postprocessing.py --control-plots --year $year $resonant \
    --data-dir $data_dir \
    --sig-samples $samples \
    --plot-dir "${MAIN_DIR}/plots/PostProcessing/$TAG" \
    --bdt-preds-dir "$data_dir/24_03_07_new_samples_max_depth_5/inferences" \
    --mass-plots
done
