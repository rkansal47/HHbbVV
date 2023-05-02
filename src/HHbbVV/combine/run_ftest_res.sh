#!/bin/bash

####################################################################################################
# 1) Makes datacards and workspaces for different orders of polynomials
# 2) Runs background-only fit in validation region for lowest order polynomial and GoF test (saturated model) on data
# 3) Generates toys and gets test statistics for each (-t|--goftoys)
# 4) Fits +1 order models to all 100 toys and gets test statistics (-f|--ffits)

# Author: Raghav Kansal
####################################################################################################

goftoys=0
ffits=0
seed=42
numtoys=100
templates_tag=Apr11

options=$(getopt -o "tf" --long "cardstag:,templatestag:,goftoys,ffits,numtoys:,seed:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -t|--goftoys)
            goftoys=1
            ;;
        -f|--ffits)
            ffitss=1
            ;;
        --cardstag)
            shift
            cards_tag=$1
            ;;
        --templatestag)
            shift
            templates_tag=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

echo "Arguments: goftoys=$goftoys ffits=$ffits seed=$seed numtoys=$numtoys"


####################################################################################################
# Set up fit args
####################################################################################################

templates_dir="/eos/uscms/store/user/rkansal/bbVV/templates/${templates_tag}"
sig_sample="NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-190"
cards_dir="cards/f_tests/${cards_tag}/"
mkdir -p ${cards_dir}
echo "Saving datacards to ${cards_dir}"

# these are for inside the different cards directories
dataset=data_obs
ws="./combined"
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="./outs"

# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..9}
do
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


####################################################################################################
# Making cards and workspaces for each order polynomial
####################################################################################################

for ord1 in {0..2}
do
    for ord2 in {0..2}
    do
        model_name="nTF1_${ord1}_nTF2_${ord2}"
        echo "Making Datacard for $model_name"
        python3 -u postprocessing/CreateDatacard.py --templates-dir ${templates_dir} --sig-separate \
        --resonant --model-name ${model_name} --sig-sample ${sig_sample} \
        --nTF ${ord2} ${ord1} --cards-dir ${cards_dir}

        cd ${cards_dir}/${model_name}/
        /uscms/home/rkansal/hhcombine/combine_scripts/run_blinded.sh -wbgr
        cd -
    done
done


####################################################################################################
# Generate toys for (0, 0) order
####################################################################################################

if [ $goftoys = 1 ]; then
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
    -n "Toys${toys_name}" -t $numtoys --saveToys -s $seed -v 9 2>&1 | tee $outsdir/gentoys.txt

    cd -
fi


####################################################################################################
# GoFs on generated toys for next order polynomials
####################################################################################################

if [ $ffit = 1 ]; then
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

            ulimit -s unlimited

            combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
            --setParameters ${maskunblindedargs},${setparams},r=0 \
            --freezeParameters ${freezeparams},r \
            -n Toys${toys_name} -v 9 -s 42 -t 100 --toysFile ${toys_file} 2>&1 | tee $outsdir/GoF_toys${toys_name}.txt

            cd -
        done
    done
fi