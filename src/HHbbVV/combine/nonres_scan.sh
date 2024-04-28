#!/bin/bash
# shellcheck disable=SC2043

templates_tag="24Apr9ggFScan"
sigsample="qqHH_CV_1_C2V_0_kl_1_HHbbVV"
# sigsample="HHbbVV"

if [[ $sigsample == "qqHH_CV_1_C2V_0_kl_1_HHbbVV" ]]
then
  sig_tag="vbf-sig-only"
else
  sig_tag="ggf-sig-only"
fi

for bdt_cut in 0.996 # 0.995 0.9965 # 0.9997 0.9999
do
  for txbb_cut in "MP" "HP"
  do
    if [[ $sigsample == "qqHH_CV_1_C2V_0_kl_1_HHbbVV" ]]
    then
      cutstr="ggf_txbb_MP_ggf_bdt_0.9965_vbf_txbb_${txbb_cut}_vbf_bdt_${bdt_cut}_lepton_veto_Hbb"
    else
      cutstr="ggf_txbb_${txbb_cut}_ggf_bdt_${bdt_cut}_vbf_txbb_HP_vbf_bdt_0.999_lepton_veto_Hbb"
    fi
    echo $cutstr

    python3 -u postprocessing/CreateDatacard.py --templates-dir templates/$templates_tag/$cutstr \
    --model-name $templates_tag/$sig_tag/$cutstr --no-do-jshifts --sig-sample $sigsample

    (
      cd cards/$templates_tag/$sig_tag/$cutstr || exit
      /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
    )
  done
done


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
