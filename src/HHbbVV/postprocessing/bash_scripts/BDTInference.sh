#!/bin/bash
# shellcheck disable=SC2086,SC2043

####################################################################################################
# Script for running BDT inference only
# Author: Raghav Kansal
####################################################################################################

data_dir="/ceph/cms/store/user/rkansal/bbVV/skimmer/24Feb5LPMatchingFix"


for year in 2016 2016APV 2017 2018
do
    python -u BDTPreProcessing.py --year "$year" --signal-data-dir "$data_dir" --bg-keys "" --no-data --no-save-data --inference --bdt-model bdt_models/24_04_05_k2v0_training_eqsig_vbf_vars_rm_deta.model --bdt-preds-dir "$data_dir"
done
