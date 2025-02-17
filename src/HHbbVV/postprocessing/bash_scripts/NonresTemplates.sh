#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2089,SC2090

####################################################################################################
# Script for creating nonresonant templates + BDT score control plots
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
data_dir="--data-dir $MAIN_DIR/../data/skimmer/24Mar14UpdateData"
signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Jan9UpdateLPFix"
# bdt_preds_dir="$data_dir/24_04_05_k2v0_training_eqsig_vbf_vars_rm_deta/inferences"
bdt_preds_dir="$signal_data_dir/inferences"
sig_samples=""
region="all"
TAG=""
bg_samples=""
plot_dir="--plot-dir ${MAIN_DIR}/plots/PostProcessing/$TAG"


options=$(getopt -o "" --long "sample:,region:,tag:,onlysignal,noplot" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --sample)
            shift
            sig_samples="--sig-samples $1"
            ;;
        --region)
            shift
            region=$1
            ;;
        --tag)
            shift
            TAG=$1
            ;;
        --onlysignal)
            bg_samples="--bg-keys '' --no-data"
            data_dir=""
            ;;
        --noplot)
            plot_dir=""
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

io_args="$data_dir --bdt-preds-dir $bdt_preds_dir  --template-dir templates/$TAG $sig_samples $bg_samples --signal-data-dir $signal_data_dir $plot_dir"

echo "get LP SFs first for all regions"
python -u postprocessing.py --year 2016 $io_args --lpsfs --nonres-regions $region

for year in 2016 2016APV 2017 2018
do
    python -u -W ignore postprocessing.py --year $year --templates $sig_samples $io_args --plot-shifts --nonres-regions $region --no-blinded
    # --control-plots --control-plot-vars "BDTScore" \
done
