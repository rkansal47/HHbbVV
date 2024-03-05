#!/usr/bin/env bash

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimits \
    --version "$VERSION" \
    --datacards "$Cbbww4qInject" \
    --xsec fb \
    --pois r \
    --scan-parameters kl,-30,-5,26:kl,-2,5,8:kl,10,30,21 \
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
    --save-hep-data False \
    --frozen-groups signal_norm_xsbr

    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \
    # --save-hep-data True \

    # --scan-parameters kl,-30,-6,25:kl,-6,10,33:kl,0,30,31 \
    # --scan-parameters kl,-30,-6,5:kl,-6,-5,2:kl,-3,-2,2:kl,0,4,5:kl,7,10,2:kl,14,30,5 \
