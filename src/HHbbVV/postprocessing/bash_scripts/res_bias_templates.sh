#!/bin/bash

TAG=$1

# backgrounds and data
for year in 2016APV 2016 2017 2018
do
    python -u postprocessing.py --templates --year $year --template-dir "/eos/uscms/store/user/rkansal/bbVV/templates/$TAG/" --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Feb24" --resonant --sig-samples "" --res-leading-pt 400 --res-subleading-pt 350 --res-thww-wp 0.6 --templates-name backgrounds --old-processor
done

# signals
for sample in NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190
do
    for year in 2016APV 2016 2017 2018
    do
    python -u postprocessing.py --templates --year $year --template-dir "/eos/uscms/store/user/rkansal/bbVV/templates/$TAG/" --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Feb24" --signal-data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Apr11" --resonant --sig-samples $sample --bg-keys "" --res-leading-pt 400 --res-subleading-pt 350 --res-thww-wp 0.6 --templates-name $sample --no-data
    done
done
