#!/bin/bash

mkdir -p tmp/test_outputs/

year=2017
extraargs=""
extraargs="--no-inference"

python -W ignore src/run.py --processor skimmer --year $year --samples TTbar --subsamples TTToSemiLeptonic --save-systematics --starti 0 --endi 1 $extraargs
mv "0-1.parquet" tmp/test_outputs/ttsl.parquet
mv "outfiles/0-1.pkl" tmp/test_outputs/ttsl.pkl

python -W ignore src/run.py --processor skimmer --year $year --samples HH --subsamples GluGluToHHTobbVV_node_cHHH1 --save-systematics --starti 0 --endi 1 $extraargs
mv "0-1.parquet" tmp/test_outputs/hhbbvv.parquet
mv "outfiles/0-1.pkl" tmp/test_outputs/hhbbvv.pkl

python -W ignore src/run.py --processor skimmer --year $year --samples XHY --subsamples NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 --save-systematics --starti 0 --endi 1 $extraargs
mv "0-1.parquet" tmp/test_outputs/xhy.parquet
mv "outfiles/0-1.pkl" tmp/test_outputs/xhy.pkl

python -W ignore src/run.py --processor skimmer --year $year --samples QCD --subsamples QCD_HT1000to1500 --save-systematics --starti 0 --endi 1 $extraargs
mv "0-1.parquet" tmp/test_outputs/qcd.parquet
mv "outfiles/0-1.pkl" tmp/test_outputs/qcd.pkl

python -W ignore src/run.py --processor skimmer --year $year --samples "JetHT$year" --subsamples "JetHT_Run${year}D" --save-systematics --starti 0 --endi 1 $extraargs
mv "0-1.parquet" tmp/test_outputs/data.parquet
mv "outfiles/0-1.pkl" tmp/test_outputs/data.pkl