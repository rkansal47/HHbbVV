#!/usr/bin/env bash

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimits \
    --version "$VERSION" \
    --datacards "$Cbbww4q" \
    --xsec fb \
    --pois r \
    --scan-parameters kl,-30,-12,7:kl,-7,-5,3:kl:kl,-2,7,19:kl,10,30,6 \
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
    # --use-snapshot True \  # for after unblinding
    # --Snapshot-workflow "local" \