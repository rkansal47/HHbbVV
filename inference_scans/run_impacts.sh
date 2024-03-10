#!/usr/bin/env bash

law run PlotPullsAndImpacts \
    --version "$VERSION" \
    --datacards "$Cbbww4qInject" \
    --file-types "pdf,png" \
    --pois r \
    --PullsAndImpacts-workflow "htcondor" \
    --PullsAndImpacts-tasks-per-job 10 \
    --PullsAndImpacts-custom-args="--rMin -40 --rMax 40" \
    --parameters-per-page 40 \
    --order-by-impact \
    --labels "nuisance_renames.py" \
    --skip-parameters "*dataResidual_Bin*" \
    --campaign run2 \
    --page -1 \
    --pull-range 3 \
    --unblinded $UNBLINDED \
    --remove-output 0,a,y
    # --print-command 2 \
    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \
