TAG=$1

# backgrounds and data
for year in 2016APV 2016 2017 2018
do
    python -u postprocessing.py --templates --year $year --template-dir "/eos/uscms/store/user/rkansal/bbVV/templates/$TAG/" --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Feb24" --resonant --sig-samples "" --res-txbb-wp LP MP HP --res-thww-wp 0.4 0.6 0.8 0.9 0.94 0.96 0.98 --no-do-jshifts --templates-name backgrounds --old-processor
done

# signals
for sample in NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 NMSSM_XToYHTo2W2BTo4Q2B_MX-2000_MY-125 NMSSM_XToYHTo2W2BTo4Q2B_MX-1200_MY-190 NMSSM_XToYHTo2W2BTo4Q2B_MX-900_MY-80
do
    for year in 2016APV 2016 2017 2018
    do
    python -u postprocessing.py --templates --year $year --template-dir "/eos/uscms/store/user/rkansal/bbVV/templates/$TAG/" --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Feb24" --signal-data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Apr11" --resonant --sig-samples $sample --bg-keys "" --res-txbb-wp LP MP HP --res-thww-wp 0.4 0.6 0.8 0.9 0.94 0.96 0.98 --no-do-jshifts --templates-name $sample --no-data
    done
done
