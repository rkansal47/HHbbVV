#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2206

####################################################################################################
# BDT Sculpting plots
# Author: Raghav Kansal
####################################################################################################

years=("2016APV" "2016" "2017" "2018")

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
bdt_preds_dir="$data_dir/24_04_03_k2v0_training_eqsig_vbf_vars/inferences"
TAG=""


options=$(getopt -o "" --long "year:,tag:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --year)
            shift
            years=($1)
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

for year in "${years[@]}"
do
    echo $year
    python postprocessing.py --year $year --data-dir $data_dir --bdt-preds-dir $bdt_preds_dir \
    --sig-samples GluGluToHHTobbVV_node_cHHH1 qqHH_CV_1_C2V_0_kl_1_HHbbVV --bg-keys QCD TT "Z+Jets" --no-data \
    --bdt-plots --plot-dir "$MAIN_DIR/plots/PostProcessing/$TAG"
done
