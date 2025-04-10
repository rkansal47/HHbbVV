#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2034

####################################################################################################
# Resonant datacards
# Author: Raghav Kansal
####################################################################################################

TAG=""
sig_sample=NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250
# templates_dir=/eos/uscms/store/user/rkansal/bbVV/templates/25Feb8XHYFix
templates_dir=/ceph/cms/store/user/rkansal/bbVV/templates/25Feb8XHYFix
# bg_templates_dir="templates/25Feb23ResTemplatesHbbUncs"
extraargs=""

if [[ ! -d /ceph/cms/store/user/rkansal/bbVV/templates ]]; then
  templates_dir=/eos/uscms/store/user/rkansal/bbVV/templates/25Feb8XHYFix
fi

bg_templates_dir="${templates_dir}/backgrounds"

options=$(getopt -o "" --long "tag:,extraargs:" -- "$@")
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

python3 -u postprocessing/CreateDatacard.py --templates-dir "${templates_dir}/${sig_sample}" --bg-templates-dir $bg_templates_dir --sig-separate --resonant --model-name $TAG --sig-sample "${sig_sample}" $extraargs || exit

cd cards/$TAG || exit
run_unblinded_res.sh -wbsg
