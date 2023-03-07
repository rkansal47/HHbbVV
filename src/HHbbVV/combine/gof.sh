#!/bin/bash

dataset=data_obs
cards_dir=$1
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir


echo "data GoF"
combine -M GoodnessOfFit -d ${wsm_snapshot}.root --setParameterRange r=-20,20 --algo saturated \
--setParameters mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin5=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin6=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin7=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin8=0,CMS_bbWW_boosted_ggf_qcdparam_msdbin9=0 \
--freezeParameters CMS_bbWW_boosted_ggf_qcdparam_msdbin5,CMS_bbWW_boosted_ggf_qcdparam_msdbin6,CMS_bbWW_boosted_ggf_qcdparam_msdbin7,CMS_bbWW_boosted_ggf_qcdparam_msdbin8,CMS_bbWW_boosted_ggf_qcdparam_msdbin9 \
-n pass 2&>1 | tee $outsdir/GoF_data.txt