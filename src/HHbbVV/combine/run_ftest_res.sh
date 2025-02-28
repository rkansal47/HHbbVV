#!/bin/bash
# shellcheck disable=SC2086

####################################################################################################
# 1) Makes datacards and workspaces for different orders of polynomials
# 2) Runs background-only fit in validation region for lowest order polynomial and GoF test (saturated model) on data
# 3) Runs fit diagnostics and saves shapes (-d|--dfit)
# 4) Generates toys and gets test statistics for each (-t|--goftoys)
# 5) Fits +1 order models to all 100 toys and gets test statistics (-f|--ffits)
#
# Author: Raghav Kansal
####################################################################################################

cards=0
goftoys=0
ffits=0
dfit=0
seed=42
numtoys=100
low1=0
low2=0
cardsdir="cards/f_tests"
scripts_dir="/uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine"
script="run_blinded.sh"
verbose=9

options=$(getopt -o "ctfdu" --long "scriptsdir:,cardsdir:,cardstag:,templatestag:,sigsample:,low1:,low2:,unblinded,cards,goftoys,ffits,dfit,numtoys:,seed:,verbose:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --scriptsdir)
            shift
            scripts_dir=$1
            ;;
        --cardsdir)
            shift
            cardsdir=$1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;
        -f|--ffits)
            ffits=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        -c|--cards)
            cards=1
            ;;
        -u|--unblinded)
            script="run_unblinded_res.sh"
            ;;
        --cardstag)
            shift
            cards_tag=$1
            ;;
        --templatestag)
            shift
            templates_tag=$1
            ;;
        --sigsample)
            shift
            sig_sample=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --low1)
            shift
            low1=$1
            ;;
        --low2)
            shift
            low2=$1
            ;;
        --verbose)
            shift
            verbose=$1
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

echo "Arguments: cardstag=$cards_tag templatestag=$templates_tag sigsample=$sig_sample dfit=$dfit \
goftoys=$goftoys ffits=$ffits seed=$seed numtoys=$numtoys low1=$low1 low2=$low2 verbose=$verbose"

echo "Running script: $script"

####################################################################################################
# Set up fit args
####################################################################################################

templates_dir="/eos/uscms/store/user/rkansal/bbVV/templates/${templates_tag}"
qcd_fit_dir="/uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/qcdfit22sl7"
cards_dir="$cardsdir/${cards_tag}/"
mkdir -p "${cards_dir}"
echo "Saving datacards to ${cards_dir}"

####################################################################################################
# Making cards and workspaces for each order polynomial
####################################################################################################

if [ $cards = 1 ]; then
    for ord1 in {0..3}
    do
        for ord2 in {0..3}
        do
            model_name="nTF1_${ord1}_nTF2_${ord2}"
            echo "$model_name"
            if [ ! -f "${cards_dir}/${model_name}/XHYModel.root" ]; then
                echo "Making Datacard for $model_name"

                python3 -u postprocessing/CreateDatacard.py --templates-dir "${templates_dir}/${sig_sample}" --bg-templates-dir "${templates_dir}/backgrounds" --qcd-fit-dir $qcd_fit_dir \
                --sig-separate --resonant --model-name "${model_name}" --sig-sample "${sig_sample}" \
                --nTF "${ord2}" "${ord1}" --cards-dir "${cards_dir}"
            fi

            cd "${cards_dir}/${model_name}"/ || exit
            if [ ! -f "higgsCombineData.GoodnessOfFit.mH125.root" ]; then
                ${scripts_dir}/$script -wbgr --verbose $verbose
            fi
            if [ $dfit = 1 ] && [ ! -f "FitShapes.root" ]; then
                ${scripts_dir}/$script -dr --verbose $verbose
            fi
            cd - > /dev/null || exit
        done
    done
fi

####################################################################################################
# Generate toys for ($low1, $low2) order
####################################################################################################

toys_name="${low1}${low2}"

if [ $goftoys = 1 ]; then
    model_name="nTF1_${low1}_nTF2_${low2}"
    cd "${cards_dir}/${model_name}/" || exit
    toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"

    ulimit -s unlimited

    echo "Toys for ($low1, $low2) order fit"
    ${scripts_dir}/$script -r --gentoys --toysname "${toys_name}" --seed "$seed" --numtoys "$numtoys" --verbose $verbose

    cd - || exit
fi


####################################################################################################
# GoFs on generated toys for next order polynomials
####################################################################################################

if [ $ffits = 1 ]; then
    # fit to toys from low1, low2 order
    toys_file="$(pwd)/${cards_dir}/nTF1_${low1}_nTF2_${low2}/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
    for ord1 in {0..1}
    do
        for ord2 in {0..1}
        do
            if [ "$ord1" -gt 0 ] && [ "$ord2" -gt 0 ]
            then
                break
            fi

            o1=$((low1 + ord1))
            o2=$((low2 + ord2))

            model_name="nTF1_${o1}_nTF2_${o2}"
            echo "Fits for $model_name"

            cd "${cards_dir}/${model_name}/" || exit

            ulimit -s unlimited
            ${scripts_dir}/$script -r --goftoys --toysname "${toys_name}" --seed "$seed" --toysfile "${toys_file}" --numtoys "$numtoys" --verbose $verbose

            cd - || exit
        done
    done
fi
