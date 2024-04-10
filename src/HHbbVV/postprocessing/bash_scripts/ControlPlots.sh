#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2206

####################################################################################################
# Script for Control Plots
# Author: Raghav Kansal
####################################################################################################


####################################################################################################
# Options
# --tag: Tag for the plots
# --nonresonant: Plots SM nonresonant samples only (by default plots 5 resonant samples as well)
# --nohem2d: Do not plot HEM2d for 2018
####################################################################################################

years=("2016APV" "2016" "2017" "2018")

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
bdt_preds_dir="$data_dir/24_04_03_k2v0_training_eqsig_vbf_vars/inferences"

TAG=""
resonant="--resonant"
samples="HHbbVV VBFHHbbVV NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80 NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150"
# samples="HHbbVV"
hem2d="--HEM2d"
controlplotvars=""

options=$(getopt -o "" --long "nonresonant,nohem2d,controlplotvars:,year:,tag:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --nonresonant)
            resonant=""
            samples="HHbbVV VBFHHbbVV qqHH_CV_1_C2V_0_kl_1_HHbbVV qqHH_CV_1_C2V_2_kl_1_HHbbVV"
            hem2d=""
            ;;
        --nohem2d)
            hem2d=""
            ;;
        --controlplotvars)
            shift
            controlplotvars="--control-plot-vars $1"
            ;;
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

echo "TAG: $TAG"
echo "Control plot vars: $controlplotvars"

for year in "${years[@]}"
do
    echo $year
    python -u postprocessing.py --control-plots --year $year ${resonant} ${hem2d} \
    --data-dir $data_dir \
    --sig-samples $samples \
    --bdt-preds-dir $bdt_preds_dir \
    --plot-dir "${MAIN_DIR}/plots/PostProcessing/$TAG" ${controlplotvars}
done
