#!/usr/bin/env bash
# shellcheck disable=SC2154

####################################################################################################
# Script for running HH inference 'law' commands
# 
# TODO: fill args
#
####################################################################################################

####################################################################################################
# Read options
####################################################################################################

rmoutput=0
limits_at_point=0
pois="r"
vbfargs=""
limits_1d_kl=0
limits_1d_c2v=0
impacts=0
snapshot=0
printdeps=""
cards="$Cbbww4q"
unblinded=$UNBLINDED

options=$(getopt -o "ips" --long "limpoint,limkl,limc2v,impacts,printdeps,inject,vbf,unblinded,snapshot,rmoutput:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -s|--snapshot)
            snapshot=1
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
        --vbf)
            pois="r_qqhh"
            vbfargs="--x-min 0.1 --x-max 10 --parameter-values C2V=0"
            ;;
        --unblinded)
            unblinded=True
            ;;
        -i|--inject)
            cards="$Cbbww4qInject"
            ;;
        -p|--printdeps)
            printdeps="--print-command -1"
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

# export DHI_CMS_POSTFIX="Supplementary"

common_args="--file-types pdf,png --unblinded $unblinded --version $VERSION $printdeps --remove-output $rmoutput,a,y --campaign run2 --use-snapshot True"
custom_args="--rMax 200"


if [ $snapshot = 1 ]; then
    law run Snapshot \
        --datacards $cards \
        --custom-args=$custom_args
fi


if [ $limits_at_point = 1 ]; then
    law run PlotUpperLimitsAtPoint \
        $common_args $vbfargs \
         --datacard-names bbVV \
        --multi-datacards $cards \
        --pois $pois \
        --show-parameters kl,kt,C2V,CV \
        --UpperLimits-workflow "htcondor" \
        --UpperLimits-tasks-per-job 1 \
        --x-log \
        --h-lines 1 \
        --save-hep-data True
fi


if [ $limits_1d_kl = 1 ]; then
    law run PlotUpperLimits \
        --version "$VERSION" \
        --datacards $cards \
        --xsec fb \
        --pois r \
        --scan-parameters kl,-30,-5,26:kl,-2,0,3:kl,2,5,4:kl,10,30,21 \
        --UpperLimits-workflow "htcondor" \
        --UpperLimits-tasks-per-job 1 \
        --y-log \
        --show-parameters "kt,C2V,CV" \
        --br bbww \
        --save-ranges \
        --save-hep-data False \
        --frozen-groups signal_norm_xsbr
fi

if [ $limits_1d_c2v = 1 ]; then
    law run PlotUpperLimits \
        --version "$VERSION" \
        --datacards $cards \
        --xsec fb \
        --pois r \
        --scan-parameters C2V,-1,-0.2,3:C2V,-0.1,0.5,7:C2V,0.7,1.5,5:C2V,1.6,2.1,6:C2V,2.2,3,3 \
        --UpperLimits-workflow "htcondor" \
        --UpperLimits-tasks-per-job 1 \
        --y-log \
        --show-parameters "kt,kl,CV" \
        --br bbww \
        --save-ranges \
        --save-hep-data False \
        --frozen-groups signal_norm_xsbr
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
        --Snapshot-custom-args="--rMax 200"
fi


