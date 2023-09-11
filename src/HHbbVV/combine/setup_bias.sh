#!/bin/bash

####################################################################################################
# Sets up signal injection bias tests
# TODO: add nonresonant option
#
# Author: Raghav Kansal
####################################################################################################

resonant=0
scale=1  # scale templates by this much

options=$(getopt -o "r" --long "resonant,cardstag:,templatestag:,scale:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --cardstag)
            shift
            cards_tag=$1
            ;;
        --templatestag)
            shift
            templates_tag=$1
            ;;
        --scale)
            shift
            scale=$1
            ;;
        -r|--resonant)
            resonant=1
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

cards_tag="${cards_tag}Scale${scale}"
templates_dir="/eos/uscms/store/user/rkansal/bbVV/templates/${templates_tag}"
cards_dir="cards/biastests/${cards_tag}/"
mkdir -p ${cards_dir}
echo "Saving datacards to ${cards_dir}"

for sample in NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250
do
    if [ ! -f "${cards_dir}/${sample}/mXbin9pass.txt" ]; then
        echo "Making Datacard for $sample"
        python3 -u postprocessing/CreateDatacard.py --templates-dir ${templates_dir} --sig-separate \
        --resonant --model-name $sample --sig-sample $sample --cards-dir ${cards_dir} --scale-templates $scale
    fi

    cd ${cards_dir}/${sample}/

    run_blinded.sh -rwb

    cd -
done