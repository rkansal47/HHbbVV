#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Script for creating nonresonant templates + BDT score control plots
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
bdt_preds_dir="$data_dir/24_04_05_k2v0_training_eqsig_vbf_vars_rm_deta/inferences"
TAG=""


options=$(getopt -o "" --long "tag:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
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

for year in 2016APV 2016 2017 2018
do
    python -u postprocessing.py --year $year --data-dir "$data_dir" --templates \
    --bdt-preds-dir $bdt_preds_dir \
    --plot-dir "${MAIN_DIR}/plots/PostProcessing/$TAG" \
    --template-dir "templates/$TAG" --plot-shifts --nonres-regions "vbf"
    # --control-plots --control-plot-vars "BDTScore" \
done
