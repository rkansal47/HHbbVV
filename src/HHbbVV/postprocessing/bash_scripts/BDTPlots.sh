MAIN_DIR="../../.."
TAG=23Nov7BDTSculpting

for year in 2016APV 2016 2017 2018
do
    python postprocessing.py --year $year --data-dir "$MAIN_DIR/../data/skimmer/Feb24/" --signal-data-dir "$MAIN_DIR/../data/skimmer/Jun10/" --bdt-preds-dir "$MAIN_DIR/../data/skimmer/Feb24/23_05_12_multiclass_rem_feats_3/inferences" --no-lp-sf-all-years --sig-samples GluGluToHHTobbVV_node_cHHH1 --bg-keys QCD --bdt-plots --plot-dir "$MAIN_DIR/plots/PostProcessing/$TAG"
done
