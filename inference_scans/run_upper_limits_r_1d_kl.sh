#!/usr/bin/env bash

# export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimits \
    --version "$VERSION" \
    --datacards "$Cbbww4qInject" \
    --xsec fb \
    --pois r \
    --scan-parameters kl,-30,-7,4:kl,-6,7,14:kl,15,30,4 \
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
    # --scan-parameters kl,-30,-12,7:kl,-7,-5,3:kl,-2,7,19:kl,10,30,6 \
    # --use-snapshot True \  # for (fit to data) after unblinding
<<<<<<< HEAD
    # --Snapshot-workflow "local" \
=======
    # --Snapshot-workflow "local" \
>>>>>>> cd879748e9464d27f7df38cd604d5e2ae694cde3
