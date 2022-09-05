# Example Commands

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
