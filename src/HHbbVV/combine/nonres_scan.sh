#!/bin/bash
# shellcheck disable=SC2043

# for lepton_veto in "None" "Hbb" "HH"
# do
#   cutstr="txbb_MP_bdt_0.998_lepton_veto_${lepton_veto}"
#   echo $cutstr

#   python3 -u postprocessing/CreateDatacard.py --templates-dir templates/24Mar8LeptonVetoScan/$cutstr \
#   --model-name 24Mar8LeptonVetoScan/$cutstr --no-do-jshifts --nTF 0 --only-sm

#   (
#     cd cards/24Mar8LeptonVetoScan/$cutstr || exit
#     /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
#   )
# done


templates_tag="24Apr2ggFk2v0Scan"

# for bdt_cut in 0.99 0.997 0.998 0.999
for bdt_cut in 0.9995 0.9999
do
  for txbb_cut in "HP"
  do
    cutstr="txbb_${txbb_cut}_bdt_${bdt_cut}_lepton_veto_Hbb"
    echo $cutstr

    python3 -u postprocessing/CreateDatacard.py --templates-dir templates/$templates_tag/$cutstr \
    --model-name $templates_tag/SM/$cutstr --no-do-jshifts --nTF 0 --only-sm

    (
      cd cards/$templates_tag/SM/$cutstr || exit
      /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
    )

    python3 -u postprocessing/CreateDatacard.py --templates-dir templates/$templates_tag/$cutstr \
    --model-name $templates_tag/k2v0/$cutstr --no-do-jshifts --nTF 0 --sig-sample qqHH_CV_1_C2V_0_kl_1_HHbbVV

    (
      cd cards/$templates_tag/k2v0/$cutstr || exit
      /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
    )
  done
done
