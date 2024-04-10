#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Script for creating nonresonant vbf templates scanning over variables
# Author: Raghav Kansal and Andres Nava
####################################################################################################

#MAIN_DIR="../../.."
data_dir="/ceph/cms/store/user/annava/projects/HHbbVV/24Mar5AllYears"
TAG=""
combine_dir="/home/users/annava/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit"
#datacard_dir="/home/users/annava/projects/HHbbVV/src/HHbbVV/postprocessing/cards"
postprocessing_dir="/home/users/annava/projects/HHbbVV/src/HHbbVV/postprocessing"


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

cd "$postprocessing_dir" || exit
mamba init
mamba activate bbVV

# Generating templates
for year in 2016APV 2016 2017 2018
do
    python -u postprocessing.py --year $year --data-dir "$data_dir" --templates --template-dir "templates/$TAG" --vbf --nonres-vbf-txbb-wp "HP" "0.99" "0.992" "0.994" "0.996" "0.999" --nonres-vbf-thww-wp 0.94 0.945 0.95 0.955
done

# Generating Datacards
for template_dir_path in "/home/users/annava/projects/HHbbVV/src/HHbbVV/postprocessing/templates/$TAG"/*; do
    if [ -d "$template_dir_path" ]; then
        template_folder_name=$(basename "$template_dir_path")
        #echo "Processing template in $template_dir_path"
        python CreateDatacard.py --templates-dir "$template_dir_path" --vbf --year "all" --sig-sample "qqHH_CV_1_C2V_0_kl_1_HHbbVV" --cards-dir "cards/$template_folder_name"
    fi
done

mamba deactivate


cd "$combine_dir" && \
source /cvmfs/cms.cern.ch/cmsset_default.sh && \
#eval `scramv1 runtime -sh` && \  # cmsenv equivalent in a script
cmsenv
for template_dir_path in "/home/users/annava/projects/HHbbVV/src/HHbbVV/postprocessing/templates/$TAG"/*
do
    if [ -d "$template_dir_path" ]; then
        template_folder_name=$(basename "$template_dir_path")
        echo $template_folder_name
        datacard_path="/home/users/annava/projects/HHbbVV/src/HHbbVV/postprocessing/cards/$template_folder_name/datacard.txt"

        # Check if datacard.txt exists
        if [ -f "$datacard_path" ]; then
            combine -M AsymptoticLimits --run blind -n .andresTest "$datacard_path"
            command_success=$?

            # Check if combine succeeded
            if [ "$command_success" -eq 0 ]; then
                echo "Combine executed successfully."
            else
                echo "Error: Combine execution failed."
                exit 1
            fi
        else
            echo "Datacard does not exist: $datacard_path"
        fi

        cd "$postprocessing_dir" || exit
    fi
done

# we can do a lepton veto scan with --lepton-veto "None" "Hbb" "HH" later...
