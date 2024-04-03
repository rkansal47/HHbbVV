#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Script for creating nonresonant templates scanning over variables
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
bdt_preds_dir="$data_dir/24_03_07_new_samples_max_depth_5/inferences"
TAG=""
lepton_veto=""
txbb_cut=""
bdt_cut=""

options=$(getopt -o "" --long "lveto,bdt,txbb,tag:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --lveto)
            lepton_veto="--lepton-veto None Hbb HH"
            ;;
        --bdt)
            # bdt_cut="--nonres-bdt-wp 0.99 0.997 0.998 0.999"
            bdt_cut="--nonres-bdt-wp 0.9995 0.9999 0.99999"
            ;;
        --txbb)
            txbb_cut="--nonres-txbb-wp MP HP"
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

for year in 2016APV 2016 2017 2018
do
    python -u postprocessing.py --year $year --data-dir "$data_dir" --bdt-preds-dir $bdt_preds_dir \
    --templates --template-dir "templates/$TAG" --no-do-jshifts \
    $lepton_veto $bdt_cut $txbb_cut --sig-samples HHbbVV qqHH_CV_1_C2V_0_kl_1_HHbbVV
done
