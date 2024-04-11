#!/bin/bash
# shellcheck disable=SC2086

####################################################################################################
# 1) Makes datacards and workspaces for different orders of polynomials
# 2) Runs background-only fit (Higgs mass window blinded) for lowest order polynomial and GoF test (saturated model) on data
# 3) Runs fit diagnostics and saves shapes (-d|--dfit)
# 4) Generates toys and gets test statistics for each (-t|--goftoys)
# 5) Fits +1 order models to all 100 toys and gets test statistics (-f|--ffits)
#
# Author: Raghav Kansal
####################################################################################################


goftoys=0
ffits=0
dfit=0
limits=0
seed=42
numtoys=100
order=0
sample="HHbbVV"

options=$(getopt -o "tfdlo:s:" --long "cardstag:,templatestag:,sample:,goftoys,ffits,dfit,limits,order:,numtoys:,seed:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -t|--goftoys)
            goftoys=1
            ;;
        -f|--ffits)
            ffits=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        --cardstag)
            shift
            cards_tag=$1
            ;;
        --templatestag)
            shift
            templates_tag=$1
            ;;
        -s|--sample)
            shift
            sample=$1
            ;;
        -o|--order)
            shift
            order=$1
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

echo "Arguments: cardstag=$cards_tag sample=$sample templatestag=$templates_tag dfit=$dfit \
goftoys=$goftoys ffits=$ffits order=$order seed=$seed numtoys=$numtoys"


####################################################################################################
# Set up fit args
####################################################################################################

templates_dir="/uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/postprocessing/templates/${templates_tag}"
cards_dir="cards/f_tests/${cards_tag}/"
mkdir -p "${cards_dir}"
echo "Saving datacards to ${cards_dir}"

# these are for inside the different cards directories
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="./outs"

# nonresonant args
maskunblindedargs="mask_pass=1,mask_fail=1,mask_passBlinded=0,mask_failBlinded=0"

# freeze qcd params in blinded bins
setparamsblinded=""
freezeparamsblinded=""
for bin in {5..9}
do
    setparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin}=0,"
    freezeparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin},"
done

# remove last comma
setparamsblinded=${setparamsblinded%,}
freezeparamsblinded=${freezeparamsblinded%,}


####################################################################################################
# Making cards and workspaces for each order polynomial
####################################################################################################

for ord in {0..3}
do
    model_name="nTF_${ord}"

    # create datacards if they don't already exist
    if [ ! -f "${cards_dir}/${model_name}/pass.txt" ]; then
        echo "Making Datacard for $model_name"
        python3 -u postprocessing/CreateDatacard.py --templates-dir "${templates_dir}" \
        --model-name "${model_name}" --nTF "${ord}" --cards-dir "${cards_dir}" --sig-sample $sample
    fi

    cd "${cards_dir}/${model_name}/" || exit
    echo "${cards_dir}/${model_name}/"

    # make workspace, background-only fit, GoF on data if they don't already exist
    if [ ! -f "./higgsCombineData.GoodnessOfFit.mH125.root" ]; then
        echo "Making workspace, doing b-only fit and gof on data"
        /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/run_blinded.sh -wbg
    fi

    if [ $dfit = 1 ]; then
        /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/run_blinded.sh -d
    fi

    if [ $limits = 1 ]; then
        /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/run_blinded.sh -l
    fi

    cd - || exit
done


####################################################################################################
# Generate toys for lower order
####################################################################################################

model_name="nTF_$order"
toys_name=$order
cd "${cards_dir}/${model_name}/" || exit
toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
cd - || exit

if [ $goftoys = 1 ]; then
    cd "${cards_dir}/${model_name}/" || exit

    ulimit -s unlimited

    echo "Toys for $order order fit"
    combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
    --snapshotName MultiDimFit --bypassFrequentistFit \
    --setParameters "${maskunblindedargs},${setparamsblinded},r=0" \
    --freezeParameters "${freezeparamsblinded},r" \
    -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 9 2>&1 | tee "$outsdir/gentoys.txt"

    cd - || exit
fi


####################################################################################################
# GoFs on generated toys for low and next high order polynomials
####################################################################################################

if [ $ffits = 1 ]; then
    for ord in $order $((order+1))
    do
        model_name="nTF_${ord}"
        echo "Fits for $model_name"

        cd "${cards_dir}/${model_name}/" || exit

        ulimit -s unlimited

        combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
        --setParameters "${maskunblindedargs},${setparamsblinded},r=0" \
        --freezeParameters "${freezeparamsblinded},r" \
        -n "Toys${toys_name}" -v 9 -s "$seed" -t "$numtoys" --toysFile "${toys_file}" 2>&1 | tee "$outsdir/GoF_toys${toys_name}.txt"

        cd - || exit
    done
fi
