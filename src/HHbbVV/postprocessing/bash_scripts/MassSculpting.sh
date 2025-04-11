#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2034

####################################################################################################
# Mass sculpting plots
# Author: Raghav Kansal
####################################################################################################

# signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Jan29UpdateXHYLP"
data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/24Mar14UpdateData"
signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Feb6XHY"
bg_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/24Mar6AllYearsBDTVars"
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
    python -W ignore -u postprocessing.py --mass-sculpting-plots --year $year --template-dir "templates/$TAG" --resonant \
    --data-dir "$data_dir" --bg-data-dirs "$bg_data_dir" --signal-data-dirs $signal_data_dir --bg-keys QCD --plot-dir "../../../plots/PostProcessing/$TAG" --sig-samples ""
    # --sig-samples "NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80"
done

# plots for all years are combined in PostProcessRes.ipynb
