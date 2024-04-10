#!/usr/bin/env bash
# shellcheck disable=SC2154

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimits \
    --version "$VERSION" \
    --datacards "$Cbbww4qInject" \
    --xsec fb \
    --pois r \
    --scan-parameters C2V,-1,-0.2,3:C2V,-0.1,0.5,7:C2V,0.7,1.5,5:C2V,1.6,2.1,6:C2V,2.2,3,3 \
    --UpperLimits-workflow "htcondor" \
    --UpperLimits-tasks-per-job 1 \
    --file-types "png,pdf" \
    --campaign run2 \
    --unblinded "$UNBLINDED" \
    --y-log \
    --show-parameters "kl,kt,C2V,CV" \
    --br bbww \
    --save-ranges \
    --remove-output 1,a,y \
    --save-hep-data True \
    --frozen-groups signal_norm_xsbr
    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \
