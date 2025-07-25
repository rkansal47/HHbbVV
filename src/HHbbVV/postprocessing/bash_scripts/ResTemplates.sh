#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2034

####################################################################################################
# Resonant signal templates
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
# signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Jan29UpdateXHYLP"
data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/24Mar14UpdateData"
# signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Feb6XHY"
signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Apr22PDFWeights"
bg_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/24Mar6AllYearsBDTVars"
TAG=""
extraargs=""
sample=NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250

options=$(getopt -o "" --long "tag:,extraargs:,sample:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --tag)
            shift
            TAG=$1
            ;;
        --extraargs)
            shift
            extraargs=$1
            ;;
        --sample)
            shift
            sample=$1
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
for year in 2018
do
    python -u postprocessing.py --templates --year $year --template-dir "templates/$TAG" --resonant \
    --data-dir "$data_dir" --bg-data-dirs "$bg_data_dir" --sig-samples $sample \
    --signal-data-dirs "$signal_data_dir" --plot-dir "${MAIN_DIR}/plots/PostProcessing/$TAG" $extraargs \
    --no-do-jshifts --plot-shifts
    # --do-jshifts --no-plot-shifts
done
