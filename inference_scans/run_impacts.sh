#!/usr/bin/env bash

law run PlotPullsAndImpacts \
    --version "$VERSION" \
    --multi-datacards "$Cbbww4q" \
    --datacard-names "bbVV" \
    --file-types "pdf,png" \
    --pois r \
    --PullsAndImpacts-workflow "htcondor" \
    --PullsAndImpacts-tasks-per-job 10 \
    --parameters-per-page 40 \
    --order-by-impact \
    --campaign run2 \
    --page 0 \
    --pull-range 3 \
    --unblinded $UNBLINDED \
    --remove-output 0,a,y
    # --use-snapshot True \  # for after unblinding
    # --Snapshot-workflow "local" \