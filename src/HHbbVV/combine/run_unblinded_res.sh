#!/bin/bash
# shellcheck disable=SC2086


####################################################################################################
# Script for fits
#
# 1) Combines cards and makes a workspace (--workspace / -w)
# 2) Background-only fit (--bfit / -b)
# 3) Expected asymptotic limits (--limits / -l)
# 4) Expected significance (--significance / -s)
# 5) Fit diagnostics (--dfit / -d)
# 6) GoF on data (--gofdata / -g)
# 7) GoF on toys (--goftoys / -t); a specific toy file can be specified with --toysfile
# 8) Impacts: initial fit (--impactsi / -i), per-nuisance fits (--impactsf $nuisance), collect (--impactsc $nuisances)
# 9) Bias test: run a bias test on toys (using post-fit nuisances) with expected signal strength
#    given by --bias X.
# 10) Generate toys generate toys for a given model with (--gentoys)
#
# Usage ./run_blinded.sh [-wblsdgt] [--numtoys 100] [--seed 42]
#
# Author: Raghav Kansal
####################################################################################################


####################################################################################################
# Read options
####################################################################################################

workspace=0
bfit=0
limits=0
toylimits=0
significance=0
dfit=0
dfit_asimov=0
dnll=0
resonant=0
gofdata=0
goftoys=0
impactsi=0
impactsf=0
impactsc=0
gentoys=0
seed=42
numtoys=100
toysname=""
toysfile="--snapshotName MultiDimFit --bypassFrequentistFit --saveToys --toysFrequentist"  # if no toys provided, use post-fit nuisance values from snapshot and save toys
verbose=9
bias=-1
mintol=0.1  # --cminDefaultMinimizerTolerance
# maxcalls=1000000000  # --X-rtd MINIMIZER_MaxCalls

options=$(getopt -o "wblsdrgti" --long "workspace,bfit,limits,significance,dfit,dfitasimov,toylimits,resonant,noggf,novbf,gofdata,goftoys,gentoys,dnll,impactsi,impactsf:,impactsc:,bias:,seed:,numtoys:,mintol:,toysname:,toysfile:,verbose:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -w|--workspace)
            workspace=1
            ;;
        -b|--bfit)
            bfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        --toylimits)
            toylimits=1
            ;;
        -s|--significance)
            significance=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        --dfitasimov)
            dfit_asimov=1
            ;;
        -r|--resonant)
            resonant=1
            ;;
        --noggf)
            echo ""  # dummy options to keep compatibility with run_blinded.sh
            ;;
        --novbf)
            echo ""  # dummy options to keep compatibility with run_blinded.sh
            ;;
        -g|--gofdata)
            gofdata=1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;
        --gentoys)
            gentoys=1
            ;;
        --dnll)
            dnll=1
            ;;
        -i|--impactsi)
            impactsi=1
            ;;
        --impactsf)
            shift
            impactsf=$1
            ;;
        --impactsc)
            shift
            impactsc=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --toysname)
            shift
            toysname=$1
            ;;
        --toysfile)
            shift
            toysfile="--toysFile $1"
            ;;
        --mintol)
            shift
            mintol=$1
            ;;
        --bias)
            shift
            bias=$1
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

echo "Arguments: resonant=$resonant workspace=$workspace bfit=$bfit limits=$limits \
significance=$significance dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys toysname=$toysname toysfile=$toysfile mintol=$mintol \
verbose=$verbose"

echo "Model: $(pwd)"

####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is masked off)
####################################################################################################

dataset=data_obs
cards_dir="./"
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir

# resonant args
ccargs=""
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

setparamsblinded="var{.*pass_.*mcstat.*}=0,var{.*fail_.*mcstat.*}=0"
freezeparamsblinded='var{.*pass_.*mcstat.*},var{.*fail_.*mcstat.*},var{.*xhy_mx.*},CMS_XHYbbWW_boosted_PNetHbbScaleFactors_correlated'
# freezing them here meant they weren't included in the impacts, so unfreezing now
# freezeparamsblinded="rgx{pass_.*mcstat.*},rgx{fail_.*mcstat.*}"

setparamsunblinded="rgx{.*passBlinded_.*mcstat.*}=0,rgx{.*failBlinded_.*mcstat.*}=0"
freezeparamsunblinded="rgx{.*passBlinded_.*mcstat.*},rgx{.*failBlinded_.*mcstat.*}"

# floating parameters using var{} floats a bunch of parameters which shouldn't be floated,
# so countering this inside --freezeParameters which takes priority.
# Although, practically even if those are set to "float", I didn't see them ever being fitted,
# so this is just to be extra safe.
unblindedparams="--freezeParameters ${freezeparamsunblinded},var{.*_In},var{.*__norm},var{n_exp_.*} --setParameters ${maskblindedargs},${setparamsunblinded}"

# excludeimpactparams='rgx{.*qcdparam_mXbin.*},rgx{passBlinded_.*mcstat.*},rgx{failBlinded_.*mcstat.*}'
# excludeimpactparams='rgx{.*qcdparam_mXbin.*},rgx{.*mcstat.*}'

echo -e "\n\n\n"
echo "mask args:"
echo "$maskblindedargs"

echo "set params:"
echo "$setparamsblinded"

echo "freeze params:"
echo "$freezeparamsblinded"

echo "unblinded params:"
echo "$unblindedparams"
echo -e "\n\n\n"

####################################################################################################
# Combine cards, text2workspace, fit, limits, significances, fitdiagnositcs, GoFs
####################################################################################################

# need to run this for large # of nuisances
# https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
ulimit -s unlimited

if [ $workspace = 1 ]; then
    echo "Combining cards $ccargs"
    combineCards.py $ccargs > $ws.txt

    echo "Running text2workspace"
    # text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt
    # new version got rid of -D arg??
    text2workspace.py $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt
else
    if [ ! -f "$wsm.root" ]; then
        echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
        exit 1
    fi
fi


if [ $bfit = 1 ]; then
    echo "Multidim fit"
    combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v $verbose \
    --cminDefaultMinimizerStrategy 1 --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    --setParameters "${maskblindedargs},${setparamsunblinded}"  --setParameterRanges r=-2,20 \
    --freezeParameters ${freezeparamsunblinded}  \
    -n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt
else
    if [ ! -f "higgsCombineSnapshot.MultiDimFit.mH125.root" ]; then
        echo "Background-only fit snapshot doesn't exist! Use the -b|--bfit option to run fit first. (Ignore this if you're only creating the workspace.)"
        exit 1
    fi
fi


if [ $limits = 1 ]; then
    echo "Limits"
    combine -M AsymptoticLimits -m 125 -n "" -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v $verbose \
    --saveWorkspace --saveToys --bypassFrequentistFit  --setParameterRanges r=-2,20  \
    ${unblindedparams} -s "$seed" --toysFrequentist --run blind 2>&1 | tee $outsdir/AsymptoticLimits.txt
fi


if [ $toylimits = 1 ]; then
    echo "Expected limits (MC Unblinded) using toys"
    combine -M HybridNew --LHCmode LHC-limits --saveHybridResult -m 125 -n "" -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v $verbose \
    ${unblindedparams},r=0 -s "$seed" --bypassFrequentistFit --rAbsAcc 5.0 -T 100 --clsAcc 10 \
    --floatParameters "${freezeparamsblinded},r" --toysFrequentist --expectedFromGrid 0.500 2>&1 | tee $outsdir/ToysLimits.txt

    # combine -M HybridNew --LHCmode LHC-limits --singlePoint 0 --saveHybridResult -m 125 -n "" -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v $verbose --saveToys \
    # ${unblindedparams},r=0 -s "$seed" --bypassFrequentistFit --rAbsAcc 1.0 -T 100 --clsAcc 10 \
    # --floatParameters "${freezeparamsblinded},r" --toysFrequentist 2>&1 | tee $outsdir/ToysLimitsSP.txt
fi


if [ $significance = 1 ]; then
    echo "Expected significance (MC Unblinded)"
    combine -M Significance -d ${wsm_snapshot}.root -n "" --significance -m 125 --snapshotName MultiDimFit -v $verbose \
    -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit \
    "${unblindedparams},r=1" \
    --floatParameters "${freezeparamsblinded},r" --toysFrequentist 2>&1 | tee $outsdir/Significance.txt
fi


if [ $dfit = 1 ]; then
    echo "Fit Diagnostics"
    combine -M FitDiagnostics -m 125 -d ${wsm}.root -n "" \
    ${unblindedparams} --setParameterRanges r=-2,20 \
    --cminDefaultMinimizerStrategy 1  --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    --ignoreCovWarning -v $verbose 2>&1 | tee $outsdir/FitDiagnostics.txt
    # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes \

    python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py fitDiagnostics.root -g nuisance_pulls.root --all --regex='^(?!.*mcstat)'  --vtol=0.3 --stol=0.1 --vtol2=2.0 --stol2=0.5

    echo "Fit Shapes B"
    PostFitShapesFromWorkspace --dataset "$dataset" -w ${wsm}.root --output FitShapesB.root \
    -m 125 -f fitDiagnostics.root:fit_b --postfit --print 2>&1 | tee $outsdir/FitShapes.txt

    echo "Fit Shapes S+B"
    PostFitShapesFromWorkspace --dataset "$dataset" -w ${wsm}.root --output FitShapesS.root \
    -m 125 -f fitDiagnostics.root:fit_s --postfit --print 2>&1 | tee $outsdir/FitShapes.txt
fi


if [ $dfit_asimov = 1 ]; then
    echo "Fit Diagnostics on Asimov dataset (MC Unblinded)"
    combine -M FitDiagnostics -m 125 -d ${wsm_snapshot}.root --snapshotName MultiDimFit \
    -t -1 --expectSignal=1 --toysFrequentist --bypassFrequentistFit --saveWorkspace --saveToys \
    "${unblindedparams}" --floatParameters "${freezeparamsblinded},r" \
    --cminDefaultMinimizerStrategy 1  --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    -n Asimov --ignoreCovWarning -v $verbose 2>&1 | tee $outsdir/FitDiagnosticsAsimov.txt

    combineTool.py -M ModifyDataSet ${wsm}.root:w ${wsm}_asimov.root:w:toy_asimov -d higgsCombineAsimov.FitDiagnostics.mH125.123456.root:toys/toy_asimov

    echo "Fit Shapes"
    PostFitShapesFromWorkspace --dataset toy_asimov -w ${wsm}_asimov.root --output FitShapesAsimov.root \
    -m 125 -f fitDiagnosticsAsimov.root:fit_b --postfit --print 2>&1 | tee $outsdir/FitShapesAsimov.txt
fi


if [ $gofdata = 1 ]; then
    echo "GoF on data"
    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    ${unblindedparams} -n Data -v $verbose 2>&1 | tee $outsdir/GoF_data.txt
fi


if [ $goftoys = 1 ]; then
    echo "GoF on toys"
    # snapshot and --bypassFrequentistFit and --toysFrequentist used for generating toys if toys file not provided (default value of $toysfile)
    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    ${unblindedparams} $toysfile \
    -n "Toys${toysname}" -v $verbose -s "$seed" -t "$numtoys" 2>&1 | tee "$outsdir/GoF_toys${toysname}${seed}.txt"
fi

if [ $gentoys = 1 ]; then
    echo "Generating toys"
    combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
    ${unblindedparams} -n "Toys${toysname}" -t "$numtoys" --saveToys -s "$seed" --toysFrequentist  --bypassFrequentistFit \
    -v $verbose 2>&1 | tee $outsdir/gentoys${seed}.txt
fi

if [ $impactsi = 1 ]; then
    echo "Initial fit for impacts"
    combineTool.py -M Impacts --snapshotName MultiDimFit -m 125 -n "impacts" \
    -d ${wsm_snapshot}.root --doInitialFit --toysFrequentist --robustFit 1 \
    ${unblindedparams} --setParameterRanges r=-2,20 --cminDefaultMinimizerStrategy=1 -v 1 2>&1 | tee $outsdir/Impacts_init.txt
fi


if [ "$impactsf" != 0 ]; then
    echo "Impact scan for $impactsf"
    # Impacts module cannot access parameters which were frozen in MultiDimFit, so running impacts
    # for each parameter directly using its internal command
    # (also need to do this for submitting to condor anywhere other than lxplus)
    combine -M MultiDimFit -n "_paramFit_impacts_$impactsf" --algo impact --redefineSignalPOIs r -P "$impactsf" \
    --floatOtherPOIs 1 --saveInactivePOI 1 --snapshotName MultiDimFit -d ${wsm_snapshot}.root \
    --toysFrequentist --robustFit 1 --cminDefaultMinimizerTolerance "$mintol" --setRobustFitTolerance 10 \
    ${unblindedparams} --setParameterRanges r=-2,20 --cminDefaultMinimizerStrategy=1 -v 1 -m 125 | tee "$outsdir/Impacts_$impactsf.txt"
fi


if [ "$impactsc" != 0 ]; then
    echo "Collecting impacts"
    combineTool.py -M Impacts --snapshotName MultiDimFit \
    -m 125 -n "impacts" -d ${wsm_snapshot}.root \
    --setParameters "${maskblindedargs}" \
    --named "$impactsc" \
    --setParameterRanges r=-2,20 -v 1 -o impacts.json 2>&1 | tee $outsdir/Impacts_collect.txt

    plotImpacts.py -i impacts.json -o impacts # --run blind
fi


if [ "$bias" != -1 ]; then
    echo "Bias test with bias $bias"
    # setting verbose > 0 here can lead to crazy large output files (~10-100GB!) because of getting
    # stuck in negative yield areas
    combine -M FitDiagnostics --trackParameters r --trackErrors r --justFit \
    -m 125 -n "bias${bias}" -d ${wsm_snapshot}.root --rMin "-15" --rMax 15 \
    --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --expectSignal "$bias" \
    ${unblindedparams},r=$bias --floatParameters ${freezeparamsblinded} \
    --robustFit=1 -t "$numtoys" -s "$seed" \
    --X-rtd MINIMIZER_MaxCalls=1000000 --cminDefaultMinimizerTolerance "$mintol" 2>&1 | tee "$outsdir/bias${bias}seed${seed}.txt"
fi

if [ "$dnll" != 0 ]; then
    echo "Delta NLL"
    combine -M MultiDimFit --algo grid -m 125 -n "Scan" -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v $verbose \
    --bypassFrequentistFit --toysFrequentist -t -1 --expectSignal 1 --rMin 0 --rMax 2 \
    ${unblindedparams} \
    --floatParameters "${freezeparamsblinded},r" 2>&1 | tee "$outsdir/dnll.txt"
    #  --points 21 --alignEdges 1

    plot1DScan.py "higgsCombineScan.MultiDimFit.mH125.root" -o scan
fi
