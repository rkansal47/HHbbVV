#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2034

####################################################################################################
# Nonresonant datacards
# Author: Raghav Kansal
####################################################################################################

TAG=25Mar15NonresUpdateLPSFs

python3 postprocessing/CreateDatacard.py --model-name $TAG/ggf-sig-only --templates-dir templates/25Feb6NonresMatchingFix --bg-templates-dir templates/24Aug26BDT995AllSigs --sig-separate --no-blinded --sig-sample "HHbbVV"

cd cards/$TAG/ggf-sig-only || exit
run_unblinded.sh -wbl
cd - || exit

python3 postprocessing/CreateDatacard.py --model-name $TAG/vbf-sig-only --templates-dir templates/25Feb6NonresMatchingFix --bg-templates-dir templates/24Aug26BDT995AllSigs --sig-separate --no-blinded --sig-sample "qqHH_CV_1_C2V_0_kl_1_HHbbVV"

cd cards/$TAG/vbf-sig-only || exit
run_unblinded.sh -wbl
cd - || exit
