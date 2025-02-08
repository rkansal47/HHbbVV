#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2034

####################################################################################################
# Resonant signal templates
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
# data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"
# signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Jan29UpdateXHYLP"
signal_data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/25Feb6XHY"
# signal_data_dir="/eos/uscms/store/user/rkansal/bbVV/skimmer/24Feb5LPMatchingFix"
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


sample="NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125"

# for year in 2016APV 2016 2017 2018
for sample in NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80
do
    for year in 2016APV 2016 2017 2018
    do
        python -u postprocessing.py --templates --year $year --template-dir "templates/$TAG/$sample" --signal-data-dirs "$signal_data_dir" --resonant --sig-samples $sample --bg-keys "" --no-data
    done
done
