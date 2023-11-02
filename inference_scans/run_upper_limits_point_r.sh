#!/usr/bin/env bash

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimitsAtPoint \
    --version "$VERSION" \
    --multi-datacards "$Cbbww4qInject" \
    --datacard-names "bbVV" \
    --file-types "pdf,png" \
    --pois r \
    --show-parameters kl,kt,C2V,CV \
    --UpperLimits-workflow "htcondor" \
    --UpperLimits-tasks-per-job 1 \
    --x-log \
    --campaign run2 \
    --unblinded $UNBLINDED \
    --h-lines 1 \
    --save-hep-data True \
    --remove-output 0,a,y
    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \
