#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Checking LP SF for each year
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
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
    # --sig-samples qqHH_CV_1_C2V_1_kl_2_HHbbVV --bg-keys "" --no-data \
    python -u postprocessing.py --year $year --data-dir "$data_dir" --templates \
    --no-lp-sf-all-years \
    --bdt-preds-dir "$data_dir/24_03_07_new_samples_max_depth_5/inferences" \
    --plot-dir "${MAIN_DIR}/plots/PostProcessing/$TAG" \
    --template-dir "test_templates/$TAG$year" --no-do-jshifts --vbf
done
