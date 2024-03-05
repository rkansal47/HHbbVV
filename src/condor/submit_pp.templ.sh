#!/bin/bash
# shellcheck disable=SC2154

cd HHbbVV/postprocessing/ || exit

# make dir for output
mkdir condor_templates

# run code
for sample in $samples
do
    for year in 2016 2016APV 2017 2018
    do
        python -u postprocessing.py --templates --resonant --no-data --read-sig-samples --bg-keys "" --year "$year" \
        --sig-samples "$sample" --template-dir "condor_templates/$sample" \
        --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Apr11/"
    done
done

#move output to eos
xrdcp -f condor_templates/* "$eosout"
