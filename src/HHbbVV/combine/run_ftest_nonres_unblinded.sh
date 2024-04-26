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
rmax=200
regions="ggf vbf"
sample="HHbbVV"

options=$(getopt -o "tfdlo:s:r:" --long "cardstag:,templatestag:,sample:,goftoys,ffits,dfit,limits,order:,numtoys:,seed:,region:,rmax:" -- "$@")
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
        -r|--region)
            shift
            regions=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --rmax)
            shift
            rmax=$1
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

# these are for inside the different cards directories
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

####################################################################################################
# Set up fit args
####################################################################################################

templates_dir="/uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/postprocessing/templates/${templates_tag}"
cards_dir="cards/f_tests/${cards_tag}/"
mkdir -p "${cards_dir}"
echo "Saving datacards to ${cards_dir}"

outsdir="./outs"

####################################################################################################
# Making cards and workspaces for each order polynomial
####################################################################################################

for region in $regions
do
    echo "Region: $region"
    for ord in {0..3}
    do
        model_name="nTF_${ord}"

        # create datacards if they don't already exist
        if [ ! -f "${cards_dir}/$region/${model_name}/pass$region.txt" ]; then
            echo "Making Datacard for $model_name"
            python3 -u postprocessing/CreateDatacard.py --templates-dir "${templates_dir}" \
            --model-name "${model_name}" --nTF "${ord}" --cards-dir "${cards_dir}/$region" --sig-sample $sample --no-blinded --nonres-regions $region
        fi

        cd "${cards_dir}/$region/${model_name}" || exit
        echo "${cards_dir}/$region/${model_name}"

        # make workspace, GoF on data if they don't already exist
        if [ ! -f "./higgsCombineData.GoodnessOfFit.mH125.root" ]; then
            echo "Making workspace, doing fit and gof on data"
            /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/run_unblinded.sh -wbg --rmax $rmax
        fi

        if [ $dfit = 1 ]; then
            /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/run_unblinded.sh -d --rmax $rmax
        fi

        if [ $limits = 1 ]; then
            /uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/run_unblinded.sh -l --rmax $rmax
        fi

        cd - || exit
    done
done

####################################################################################################
# Generate toys for lower order
####################################################################################################

model_name="nTF_$order"
toys_name=$order
cd "${cards_dir}/$region/${model_name}/" || exit
toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
cd - || exit

if [ $goftoys = 1 ]; then
    cd "${cards_dir}/$region/${model_name}/" || exit

    ulimit -s unlimited

    echo "Toys for $order order fit"
    combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root --rMax $rmax \
    --snapshotName MultiDimFit --bypassFrequentistFit --trackParameters r \
    -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 9 2>&1 | tee "$outsdir/gentoys.txt"

    # combine ${wsm_snapshot}.root -M MultiDimFit -m 125 --rMax $rmax \
    # --snapshotName MultiDimFit --bypassFrequentistFit --trackParameters r --floatParameters r \
    # -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 9 2>&1 | tee "$outsdir/gentoys.txt"

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

        combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 --rMax $rmax \
        -n "Toys${toys_name}" -v 9 -s "$seed" -t "$numtoys" --toysFile "${toys_file}" 2>&1 | tee "$outsdir/GoF_toys${toys_name}.txt"

        cd - || exit
    done
fi
