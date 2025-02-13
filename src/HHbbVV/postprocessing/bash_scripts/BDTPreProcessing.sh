#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Script for creating BDT training / inference datasets
# Author: Raghav Kansal
####################################################################################################

MAIN_DIR="../../.."
data_dir="$MAIN_DIR/../data/skimmer/24Mar14UpdateData"

for year in 2016APV 2016 2017 2018
do
    python -u BDTPreProcessing.py --year $year --data-dir "$data_dir"
done
