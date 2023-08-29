#!/usr/bin/env bash

law run PlotPullsAndImpacts \
    --version "$VERSION" \
    --datacards "$Cbbww4q" \
    --file-types "pdf,png" \
    --pois r \
    --PullsAndImpacts-workflow "htcondor" \
    --PullsAndImpacts-tasks-per-job 10 \
    --parameters-per-page 40 \
    --order-by-impact \
    --labels "nuisance_renames.py" \
    --skip-parameters "*dataResidual_Bin*" \
    --left-margin 500 \
    --campaign run2 \
    --page 0 \
    --pull-range 3 \
    --unblinded $UNBLINDED \
    --remove-output 0,a,y
    # --use-snapshot True \  # for (fit to data) after unblinding
    # --Snapshot-workflow "local" \