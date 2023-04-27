#!/bin/bash

# 1) Makes datacards for different orders of polynomials
# 2) Runs background-only fit in validation region for lowest order polynomial and GoF test (saturated model) on data
# 3) Generates 100 toys and gets test statistics for each
# 4) Fits +1 order models to all 100 toys and gets test statistics
# Author: Raghav Kansal

templates_tag=Apr11
templates_dir="/eos/uscms/store/user/rkansal/bbVV/templates/${templates_tag}"
sig_sample="NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-190"
cards_dir="cards/f_tests/$1/"
mkdir -p ${cards_dir}
echo "Saving datacards to ${cards_dir}"

# these are for inside the different cards directories
dataset=data_obs
ws="./combined"
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="./outs"

# one channel per bin
ccargs=""
# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..9}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=mXbin${bin}${channel}.txt "
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


####################################################################
# Making cards and workspaces for each order polynomial
####################################################################

# for ord1 in {0..2}
# do
#     for ord2 in {0..2}
#     do
#         model_name="nTF1_${ord1}_nTF2_${ord2}"
#         echo "Making Datacard for $model_name"
#         python3 -u postprocessing/CreateDatacard.py --templates-dir ${templates_dir} --sig-separate \
#         --resonant --model-name ${model_name} --sig-sample ${sig_sample} \
#         --nTF1 ${ord1} --nTF2 ${ord2} --cards-dir ${cards_dir}

#         cd ${cards_dir}/${model_name}/
#         mkdir -p $outsdir
#         pwd

#         # need to run this for large # of nuisances 
#         # https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
#         ulimit -s unlimited

#         echo "combine cards"
#         combineCards.py $ccargs > $ws.txt

#         echo "text2workspace"
#         text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt

#         echo "Blinded bkg-only fit"
#         combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v 9 --cminDefaultMinimizerStrategy 1 \
#         --setParameters ${maskunblindedargs},${setparams},r=0  \
#         --freezeParameters r,${freezeparams} \
#         -n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt

#         echo "GoF on data"
#         combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
#         --snapshotName MultiDimFit --bypassFrequentistFit \
#         --setParameters ${maskunblindedargs},${setparams},r=0 \
#         --freezeParameters ${freezeparams},r \
#         -n Data -v 9 2>&1 | tee $outsdir/GoF_data.txt

#         cd -
#     done
# done

####################################################################
# Fit for lowest order polynomial, get GoF on data and generate toys
####################################################################

model_name="nTF1_0_nTF2_0"
toys_name="00"
cd ${cards_dir}/${model_name}/
toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.42.root"

ulimit -s unlimited

echo "Toys for (0, 0) order fit"
combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
--snapshotName MultiDimFit --bypassFrequentistFit \
--setParameters ${maskunblindedargs},${setparams},r=0 \
--freezeParameters ${freezeparams},r \
-n "Toys${toys_name}" -t 100 --saveToys -s 42 -v 9 2>&1 | tee $outsdir/gentoys.txt

cd -

####################################################################
# GoFs on generated toys for next order polynomials
####################################################################

for ord1 in {0..1}
do
    for ord2 in {0..1}
    do
        if [ $ord1 -gt 0 ] && [ $ord2 -gt 0 ]
        then
            break
        fi

        model_name="nTF1_${ord1}_nTF2_${ord2}"
        echo "Fits for $model_name"

        cd ${cards_dir}/${model_name}/

        # need to run this for large # of nuisances 
        # https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
        ulimit -s unlimited

        combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
        --setParameters ${maskunblindedargs},${setparams},r=0 \
        --freezeParameters ${freezeparams},r \
        -n Toys${toys_name} -v 9 -s 42 -t 100 --toysFile ${toys_file} 2>&1 | tee $outsdir/GoF_toys${toys_name}.txt

        cd -
    done
done