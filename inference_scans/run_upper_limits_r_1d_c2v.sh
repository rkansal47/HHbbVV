#!/usr/bin/env bash

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimits \
    --version "$VERSION" \
    --datacards "$Cbbww4q" \
    --xsec fb \
    --pois r \
    --scan-parameters C2V,-1,3,10 \
    --UpperLimits-workflow "htcondor" \
    --UpperLimits-tasks-per-job 1 \
    --file-types "png,pdf" \
    --campaign run2 \
    --unblinded $UNBLINDED \
    --y-log \
    --show-parameters "kt,C2V,CV" \
    --br bbww \
    --save-ranges \
    --remove-output 0,a,y \
    --save-hep-data True \
    --frozen-groups signal_norm_xsbr
    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \
    # --scan-parameters kl,-3,5,8 \