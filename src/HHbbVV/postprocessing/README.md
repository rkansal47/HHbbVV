# Example Commands

## BDT Trainings

```bash
python TrainBDT.py --model-dir testBDT --use-sample-weights --equalize-weights --absolute-weights --data-path "../../../../data/skimmer/Feb20/bdt_data.parquet"
```

## PostProcessing

```bash
python PostProcess.py --templates \
--template-file "templates/bdtcut_0.986_bbcut_0.976.pkl" \
--plot-dir "../../../plots/PostProcess/09_02/" \
--bdt-preds "../../../../data/skimmer/Apr28/absolute_weights_preds.npy" \
--data-dir "../../../../data/skimmer/Apr28/" 
```

## PlotFits

```bash
python PlotFits.py --fit-file "cards/test_tied_stats/fitDiagnosticsBlindedBkgOnly.root" --plots-dir "../../../plots/PostFit/09_02/"
```

## CreateDatacard.py

Need `root==6.22.6`, and `square_coef` branch of https://github.com/rkansal47/rhalphalib installed (`pip install -e . --user` after checking out the branch). `CMSSW_11_2_0` recommended.

```bash
python3 postprocessing/CreateDatacard.py --templates-file templates/Jan31/templates.pkl --model-name Jan31
```

