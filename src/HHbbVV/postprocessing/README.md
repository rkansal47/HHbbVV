# Example Commands



## BDT Trainings

```bash
python TrainBDT.py --model-dir testBDT --use-sample-weights --equalize-weights --absolute-weights --data-path "../../../../data/skimmer/Feb20/bdt_data.parquet"
```

## PostProcessing

```bash
python postprocessing.py --templates --control-plots --year "2017" --template-file "templates/$TAG.pkl" --plot-dir "../../../plots/PostProcessing/$TAG/" --data-dir "../../../../data/skimmer/Feb20/"
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

