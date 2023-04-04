#!/bin/bash

dataset=data_obs
cards_dir="./"
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir

maskunblindedargs="mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0"
maskblindedargs="mask_pass=0,mask_fail=0,mask_passBlinded=1,mask_failBlinded=1"

# freeze qcd params in blinded bins
setqcdparams=""
freezeqcdparams=""
for bin in {5..9}
do
    setqcdparams+="CMS_bbWW_boosted_ggf_qcdparam_msdbin${bin}=0,"
    freezeqcdparams+="CMS_bbWW_boosted_ggf_qcdparam_msdbin${bin},"
done

# remove last comma
setqcdparams=${setqcdparams%,}
freezeqcdparams=${freezeqcdparams%,}

echo "GoF on data"
combine -M GoodnessOfFit -d ${wsm_snapshot}.root --setParameterRange r=-20,20 --algo saturated -m 125 \
--setParameters ${maskunblindedargs},${setqcdparams} \
--freezeParameters ${freezeqcdparams} \
-n passData -v 9 2>&1 | tee $outsdir/GoF_data.txt

echo "GoF on toys"
combine -M GoodnessOfFit -d ${wsm_snapshot}.root --setParameterRange r=-20,20 --algo saturated -m 125 \
--setParameters ${maskunblindedargs},${setqcdparams} \
--freezeParameters ${freezeqcdparams} \
-n passToys -v 9 -s 42 -t 100 --toysFrequentist 2>&1 | tee $outsdir/GoF_toys.txt