#!/bin/bash
# shellcheck disable=SC2043

for lepton_veto in "None" "Hbb" "HH"
do
  cutstr="txbb_MP_bdt_0.998_lepton_veto_${lepton_veto}"
  echo $cutstr

  python3 -u postprocessing/CreateDatacard.py --templates-dir templates/24Mar8LeptonVetoScan/$cutstr \
  --model-name 24Mar8LeptonVetoScan/$cutstr --no-do-jshifts --nTF 0 --only-sm

  (
    cd cards/24Mar8LeptonVetoScan/$cutstr || exit
    /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
  )
done
