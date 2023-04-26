#!/bin/bash

# Makes datacards for different orders of polynomials
# and runs background-only fit in validation region and GoF test (saturated model) on data for each
# Then GoF for 100 toys for the lowest order (0, 0)
# Author: Raghav Kansal

templates_tag=Apr11
templates_dir="/eos/uscms/store/user/rkansal/bbVV/templates/${templates_tag}"
sig_sample="NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-190"
cards_dir="cards/f_tests/$1/"
mkdir -p ${cards_dir}
echo "Saving datacards to ${cards_dir}"

# these are for inside the different cards directories
ws="./combined"
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="./outs"
mkdir -p $outsdir

# one channel per bin
ccargs=""
# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..9}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=${cards_dir}/mXbin${bin}${channel}.txt "
    done

    for channel in fail pass;
    do
        maskunblindedargs+="mask_mXbin${bin}${channel}=1,mask_mXbin${bin}${channel}Blinded=0,"
        maskblindedargs+="mask_mXbin${bin}${channel}=0,mask_mXbin${bin}${channel}Blinded=1,"
    done
done

# remove last comma
maskunblindedargs=${maskunblindedargs%,}
maskblindedargs=${maskblindedargs%,}

setparams="rgx{pass_.*mcstat.*}=0,rgx{fail_.*mcstat.*}=0"
freezeparams="rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*},rgx{.*xhy_mx.*}"

echo "mask unblinded args:"
echo $maskunblindedargs


for ord1 in {0..2}
do
    for ord2 in {0..2}
    do
        model_name="nTF1_${ord1}_nTF2_${ord2}"
        echo "Making Datacard for $model_name"
        python3 postprocessing/CreateDatacard.py --templates-dir ${templates_dir} --sig-separate \
        --resonant --model-name ${model_name} --sig-sample ${sig_sample} \
        --nTF1 ${ord1} --nTF2 ${ord2} --cards-dir ${cards_dir}

        cd ${cards_dir}/${model_name}/

        # need to run this for large # of nuisances 
        # https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
        ulimit -s unlimited

        echo "combine cards"
        combineCards.py $ccargs > $ws.txt

        echo "text2workspace"
        text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt

        echo "blinded bkg-only fit snapshot"
        combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v 9 --cminDefaultMinimizerStrategy 1 \
        --setParameters ${maskunblindedargs},${setparams},r=0  \
        --freezeParameters r,${freezeparams} \
        -n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt

        echo "GoF on data"
        combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
        --setParameters ${maskunblindedargs},${setparams},r=0 \
        --freezeParameters ${freezeparams},r \
        -n passData -v 9 2>&1 | tee $outsdir/GoF_data.txt

        echo "Finished with ${model_name}"
        cd -
    done
done


echo "Toys for (0, 0) order fit"
model_name="nTF1_0_nTF2_0"
cd ${cards_dir}/${model_name}/

combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
--setParameters ${maskunblindedargs},${setparams},r=0 \
--freezeParameters ${freezeparams},r \
-n passToys -v 9 -s 42 -t 100 --toysFrequentist 2>&1 | tee $outsdir/GoF_toys.txt