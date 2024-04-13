#!/usr/bin/env bash
# shellcheck disable=SC2154

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimitsAtPoint \
    --version "$VERSION" \
    --multi-datacards "$Cbbww4qInject" \
    --datacard-names "bbVV" \
    --file-types "pdf,png" \
    --pois r_qqhh \
    --show-parameters kl,kt,C2V,CV \
    --parameter-values C2V=0 \
    --UpperLimits-workflow "htcondor" \
    --UpperLimits-tasks-per-job 1 \
    --x-log \
    --x-min 0.1 \
    --x-max 10 \
    --campaign run2 \
    --unblinded "$UNBLINDED" \
    --h-lines 1 \
    --save-hep-data True \
    --remove-output 0,a,y
    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \
