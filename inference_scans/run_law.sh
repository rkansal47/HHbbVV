#!/usr/bin/env bash
# shellcheck disable=SC2154,SC2086,SC2153

####################################################################################################
# Script for running HH inference 'law' commands
#
# Usage:
# 0) snapshot: --snapshot
# 1) limits at point: --limpoint (--vbf)  # --vbf runs it for the VBF k2v=0 point
# 2) 1D kL limit scan: --limkl
# 3) 1D k2V limit scan: --limc2v
# 4) impacts: --impacts  (should replot the merged json with plotImpacts.py)
# 5) fit diagnostics: --fitd
#
# General options:
#   --unblinded:    unblinded
#   --printdeps:    to print dependencies only
#   --inject:       inject parameters from preliminary fit
#   --rmoutput X:   remove outputs up to depth X
#
####################################################################################################

####################################################################################################
# Read options
####################################################################################################

rmoutput=0
fitd=0
limits_at_point=0
pois="r"
vbfargs=""
limits_1d_kl=0
limits_1d_c2v=0
impacts=0
snapshot=0
gof=0
printdeps=""
cards="$Cbbww4q"
unblinded=$UNBLINDED
c2vscan="C2V,-1,3,41"
# c2vscan="C2V"
# klscan="kl,-9,-6,2:kl,-6,-5,2:kl,-1,0,2:kl,2,5,4:kl,10,11,2"
klscan="kl,-8,10,19"
cl=0.95
workflow="htcondor"

options=$(getopt -o "ips" --long "fitd,limpoint,limkl,limc2v,impacts,gof,printdeps,printcommands,inject,vbf,noggf,novbf,unblinded,snapshot,rmoutput:,cl:,workflow:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -s|--snapshot)
            snapshot=1
            ;;
        --fitd)
            fitd=1
            ;;
        --limpoint)
            limits_at_point=1
            ;;
        --limkl)
            limits_1d_kl=1
            ;;
        --limc2v)
            limits_1d_c2v=1
            ;;
        --impacts)
            impacts=1
            ;;
        --gof)
            gof=1
            ;;
        --vbf)
            pois="r_qqhh"
            vbfargs="--x-min 0.1 --x-max 10 --parameter-values C2V=0"
            ;;
        --noggf)
            pois="r_qqhh"
            # c2vscan="C2V,-1,-0.2,3:C2V,-0.1,0.3,5:C2V,1.0,1.2,2:C2V,1.2,1.5,2:C2V,1.7,2.1,5:C2V,2.2,3,3"
            ;;
        --novbf)
            pois="r_gghh"
            ;;
        --unblinded)
            unblinded=True
            ;;
        -i|--inject)
            cards="$Cbbww4qInject"
            ;;
        -p|--printdeps)
            printdeps="--print-deps -1"
            ;;
        --printcommands)
            printdeps="--print-command -1"
            ;;
        --cl)
            shift
            cl=$1
            ;;
        --workflow)
            shift
            workflow=$1
            ;;
        --rmoutput)
            shift
            rmoutput=$1
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

export DHI_CMS_POSTFIX=""

common_args="--file-types pdf,png --unblinded $unblinded --version $VERSION $printdeps --remove-output $rmoutput,a,y --campaign run2 --use-snapshot False"
custom_args="--rMax 200 --setParameterRanges r_qqhh=-40,1000:r_gghh=-40,200"


if [ $snapshot = 1 ]; then
    law run Snapshot \
        --version $VERSION $printdeps --remove-output $rmoutput,a,y --datacards $cards --parameter-values C2V=0.6 --unblinded $unblinded \
        --pois $pois \
        --custom-args="$custom_args" \
        --workflow "local"
fi

if [ $fitd = 1 ]; then
    law run PlotPostfitSOverB \
        $common_args \
        --datacards $cards \
        --campaign run2 \
        --pois $pois \
        --parameter-values C2V=0.6:CV=1:kl=1:kt=1 \
        --frozen-parameters C2V,CV,kl,kt \
        --workflow "local"
fi


if [ $limits_at_point = 1 ]; then
    law run PlotUpperLimitsAtPoint \
        $common_args $vbfargs \
        --datacard-names bbVV \
        --multi-datacards $cards \
        --pois $pois \
        --show-parameters kl,kt,C2V,CV \
        --UpperLimits-workflow $workflow \
        --UpperLimits-tasks-per-job 1 \
        --UpperLimits-custom-args="--cl $cl --verbose 1 --cminDefaultMinimizerStrategy 0 --cminPoiOnlyFit --freezeParameters r,r_gghh,kl,kt,CV,C2V" \
        --x-log \
        --h-lines 1 \
        --save-hep-data True \
        --Snapshot-custom-args="$custom_args"
        # --frozen-groups signal_norm_xsbr
fi


if [ $limits_1d_kl = 1 ]; then
    law run PlotUpperLimits \
        $common_args $vbfargs \
        --version "$VERSION" \
        --datacards $cards \
        --xsec fb \
        --pois $pois \
        --scan-parameters $klscan \
        --x-min -9 --x-max 11 \
        --UpperLimits-workflow "htcondor" \
        --UpperLimits-tasks-per-job 1 \
        --y-log \
        --show-parameters "kt,C2V,CV" \
        --br bbww \
        --save-ranges \
        --save-hep-data False \
        --Snapshot-custom-args="$custom_args"
        # --frozen-groups signal_norm_xsbr
fi

if [ $limits_1d_c2v = 1 ]; then
    law run PlotUpperLimits \
        $common_args $vbfargs \
        --version "$VERSION" \
        --datacards $cards \
        --xsec fb \
        --pois $pois \
        --scan-parameters $c2vscan \
        --UpperLimits-workflow "htcondor" \
        --UpperLimits-tasks-per-job 1 \
        --UpperLimits-custom-args="--cminDefaultMinimizerStrategy 0" \
        --y-log \
        --show-parameters "kt,kl,CV" \
        --br bbww \
        --save-ranges \
        --save-hep-data False \
        --Snapshot-custom-args="$custom_args"
        # --frozen-groups signal_norm_xsbr
fi


if [ $impacts = 1 ]; then
    law run PlotPullsAndImpacts \
        $common_args \
        --datacards $cards \
        --pois r \
        --PullsAndImpacts-workflow "htcondor" \
        --PullsAndImpacts-tasks-per-job 10 \
        --PullsAndImpacts-custom-args="--rMin -40 --rMax 200" \
        --parameters-per-page 40 \
        --order-by-impact \
        --labels "nuisance_renames.py" \
        --skip-parameters "*dataResidual_Bin*" \
        --page -1 \
        --pull-range 3 \
        --Snapshot-custom-args="$custom_args"
fi


if [ $gof = 1 ]; then
    law run PlotGoodnessOfFit \
        --file-types pdf,png --version $VERSION $printdeps --remove-output $rmoutput,a,y --campaign run2 --use-snapshot False \
        $vbfargs \
        --datacards $cards \
        --pois $pois \
        --show-parameters kl,kt,C2V,CV \
        --toys 500 \
        --toys-per-branch 20 \
        --frequentist-toys \
        --GoodnessOfFit-workflow $workflow \
        --Snapshot-custom-args="$custom_args"
        # --frozen-groups signal_norm_xsbr
fi
