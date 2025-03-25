#!/bin/bash
# shellcheck disable=SC2086

numtoys=100
seed=42
dofitd=0

# create model and do goodness of fits
for ord1 in {0..3}
do
    for ord2 in {0..3}
    do
        model_name="nTF${ord1}${ord2}"
        toys_name="${ord1}${ord2}"
        echo "$model_name"

        if [ ! -f "${model_name}/qcdmodel.root" ]; then
            echo "Making Datacard for $model_name"
            python3 ../../postprocessing/CreateQCDModel.py --nTF "${ord1}" "${ord2}"
        fi

        cd "${model_name}"/ || exit
        mkdir -p outs

        if [ ! -f "higgsCombineData.GoodnessOfFit.mH125.root" ]; then
            echo "Workspace and GoF"
            chmod u+x build.sh
            ./build.sh
            combine -M MultiDimFit -d model_combined.root -m 125 --saveWorkspace --setParameters r=0 --freezeParameters r -n "Snapshot" -v5 2>&1 | tee outs/MultiDimFit.txt
            combine -M GoodnessOfFit -d higgsCombineSnapshot.MultiDimFit.mH125.root --snapshotName MultiDimFit --algo saturated --freezeParameters r --setParameters r=0 -n Data -m 125 -v2 2>&1 | tee outs/GoF_data.txt
        fi

        if [ $dofitd = 1 ] && [ ! -f "FitShapesB.root" ]; then
            echo "Fit diagnostics and shapes"
            combine -M FitDiagnostics -d model_combined.root -m 125 --setParameters r=0 --freezeParameters r -n ""
            PostFitShapesFromWorkspace -w model_combined.root --output FitShapesB.root -m 125 -f fitDiagnostics.root:fit_b --postfit
        fi

        cd - > /dev/null || exit
    done
done


for ord1 in {0..2}
do
    for ord2 in {0..2}
    do

        if [ "$ord1" == 2 ] && [ "$ord2" == 0 ]
        then
            continue
        fi


        echo -e "\n\n\n"
        model_name="nTF${ord1}${ord2}"
        toys_name="${ord1}${ord2}"
        cd "${model_name}"/ || exit

        toys_file="$(pwd)/higgsCombineToys.GenerateOnly.mH125.$seed.root"

        # echo "Generating toys for ($ord1, $ord2) order"
        # combine -M GenerateOnly -m 125 -d higgsCombineSnapshot.MultiDimFit.mH125.root --freezeParameters r --setParameters r=0 --toysFrequentist --bypassFrequentistFit -n Toys${toysname} --snapshotName MultiDimFit --saveToys -t $numtoys -s "$seed" -v2 2>&1 | tee outs/GenerateOnly.txt

        echo "Fitting to toys from ($ord1, $ord2) order"
        # combine -M GoodnessOfFit -m 125 -d higgsCombineSnapshot.MultiDimFit.mH125.root --algo saturated --freezeParameters r --setParameters r=0 --toysFrequentist --bypassFrequentistFit --toysFile $toys_file -n Toys$toys_name -t $numtoys 2>&1 | tee outs/GoF_toys$toys_name.txt

        cd - > /dev/null || exit

        # fit higher order models to these toys
        for high1 in {0..1}
        do
            for high2 in {0..1}
            do
                if [ "$high1" == 0 ] && [ "$high2" == 0 ]
                then
                    continue
                fi

                if [ "$high1" == 1 ] && [ "$high2" == 1 ]
                then
                    break
                fi

                o1=$((high1 + ord1))
                o2=$((high2 + ord2))

                echo -e "\n\n\n"
                model_name="nTF${o1}${o2}"
                echo "Fits for $model_name"

                cd "${model_name}"/ || exit

                # change to actual seed!
                if [ ! -f "higgsCombineToys$toys_name.GoodnessOfFit.mH125.123456.root" ]; then
                    echo "Running fits"
                    combine -M GoodnessOfFit -m 125 -d higgsCombineSnapshot.MultiDimFit.mH125.root --algo saturated --freezeParameters r --setParameters r=0 --toysFrequentist --bypassFrequentistFit --toysFile $toys_file -n Toys$toys_name -t $numtoys 2>&1 | tee outs/GoF_toys$toys_name.txt
                fi

                cd - > /dev/null || exit
            done
        done
    done
done
