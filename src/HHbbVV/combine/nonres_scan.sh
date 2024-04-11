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

templates_tag="24Apr9ggFScan"
nTF=0
# sigsample="qqHH_CV_1_C2V_0_kl_1_HHbbVV"
sigsample="HHbbVV"

for bdt_cut in 0.996 #m0.995 0.9965 # 0.9997 0.9999
do
  for txbb_cut in "MP" "HP"
  do
    cutstr="ggf_txbb_${txbb_cut}_ggf_bdt_${bdt_cut}_vbf_txbb_HP_vbf_bdt_0.999_lepton_veto_Hbb"
    echo $cutstr

    # python3 -u postprocessing/CreateDatacard.py --templates-dir templates/$templates_tag/$cutstr \
    # --model-name $templates_tag/SM/$cutstr --no-do-jshifts --nTF 0 --only-sm

    # (
    #   cd cards/$templates_tag/SM/$cutstr || exit
    #   /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
    # )

    model_name=${templates_tag}_nTF$nTF/$cutstr
    python3 -u postprocessing/CreateDatacard.py --templates-dir templates/$templates_tag/$cutstr \
    --model-name $model_name --no-do-jshifts --nTF $nTF --sig-sample $sigsample

    (
      cd cards/$model_name || exit
      /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbl
    )
  done
done
