#!/bin/bash
# shellcheck disable=SC2086

# for f in hhbbvv xhy ttsl qcd data
# do
#     mkdir -p tmp/test_outputs/2017/$f/parquet tmp/test_outputs/2017/$f/pickles
#     mv tmp/test_outputs/2017/$f.parquet tmp/test_outputs/2017/$f/parquet/
#     mv tmp/test_outputs/2017/$f.pkl tmp/test_outputs/2017/$f/pickles/
# done


year=2018
processor=ttsfs
extraargs=""
# extraargs="--no-inference"

OUTPUTDIR="tmp/test_outputs_ttsfs/$year"
mkdir -p $OUTPUTDIR

# python -W ignore src/run.py --processor $processor --year $year --samples HH --subsamples GluGluToHHTobbVV_node_cHHH1 --save-systematics --starti 0 --endi 1 $extraargs
# label="GluGluToHHTobbVV_node_cHHH1"
# mkdir -p $OUTPUTDIR/$label/parquet $OUTPUTDIR/$label/pickles
# mv "0-1.parquet" $OUTPUTDIR/$label/parquet/
# mv "outfiles/0-1.pkl" $OUTPUTDIR/$label/pickles/

# python -W ignore src/run.py --processor $processor --year $year --samples XHY --subsamples NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250 --save-systematics --starti 0 --endi 1 $extraargs
# label="NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250"
# mkdir -p $OUTPUTDIR/$label/parquet $OUTPUTDIR/$label/pickles
# mv "0-1.parquet" $OUTPUTDIR/$label/parquet/
# mv "outfiles/0-1.pkl" $OUTPUTDIR/$label/pickles/

python -W ignore src/run.py --processor $processor --year $year --samples TTbar --subsamples TTToSemiLeptonic --starti 0 --endi 1 $extraargs
label="TTToSemiLeptonic"
mkdir -p $OUTPUTDIR/$label/parquet $OUTPUTDIR/$label/pickles
mv "0-1.parquet" $OUTPUTDIR/$label/parquet/
mv "outfiles/0-1.pkl" $OUTPUTDIR/$label/pickles/

python -W ignore src/run.py --processor $processor --year $year --samples SingleTop --subsamples ST_t-channel_top_4f_InclusiveDecays --starti 0 --endi 1 $extraargs
label="ST_t-channel_top_4f_InclusiveDecays"
mkdir -p $OUTPUTDIR/$label/parquet $OUTPUTDIR/$label/pickles
mv "0-1.parquet" $OUTPUTDIR/$label/parquet/
mv "outfiles/0-1.pkl" $OUTPUTDIR/$label/pickles/

python -W ignore src/run.py --processor $processor --year $year --samples QCD --subsamples QCD_HT1000to1500 --save-systematics --starti 0 --endi 1 $extraargs
label="QCD_HT1000to1500"
mkdir -p $OUTPUTDIR/$label/parquet $OUTPUTDIR/$label/pickles
mv "0-1.parquet" $OUTPUTDIR/$label/parquet/
mv "outfiles/0-1.pkl" $OUTPUTDIR/$label/pickles/

python -W ignore src/run.py --processor $processor --year $year --samples "SingleMu$year" --subsamples "SingleMuon_Run${year}D" --save-systematics --starti 0 --endi 1 $extraargs
label="SingleMuon_Run${year}D"
mkdir -p $OUTPUTDIR/$label/parquet $OUTPUTDIR/$label/pickles
mv "0-1.parquet" $OUTPUTDIR/$label/parquet/
mv "outfiles/0-1.pkl" $OUTPUTDIR/$label/pickles/
