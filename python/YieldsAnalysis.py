"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""


import utils
import plotting
import matplotlib.pyplot as plt
from coffea.processor import PackedSelection
import numpy as np

import importlib
importlib.reload(utils)
importlib.reload(plotting)


plotdir = "../plots/YieldsAnalysis/Sep15/"

import os
os.system(f"mkdir -p {plotdir}")


events = utils.load_events()

for key in utils.getAllKeys():
    print(f"Preselecting {key} events")
    cut = (events[key]["bbFatJetParticleNetMD_Txbb"] > 0.8) * (events[key]["bbFatJetMsd"] > 50) * (events[key]["VVFatJetMsd"] > 50)
    for var in events[key].keys():
        events[key][var] = events[key][var][cut]


CUT_MAX_VAL = 9999

final_mass_sig_region = [100, 150]

##################################################################################
# Cut-based selection (optimized for AK15 only; TODO: optimize for hybrid mode)
##################################################################################

cut_based_sel_no_bb_mass_cuts = {
    'bbFatJetPt': [250, CUT_MAX_VAL],
    'VVFatJetPt': [250, CUT_MAX_VAL],
    # 'bbFatJetMsd': [100, 150],
    'VVFatJetMsd': [110, 150],
    'bbFatJetParticleNetMD_Txbb': [0.98, CUT_MAX_VAL],
    'VVFatJetParticleNet_Th4q': [0.9, CUT_MAX_VAL],
}

cut_based_no_bb_mass_selection, cut_based_no_bb_mass_cutflow = utils.make_selection(cut_based_sel_no_bb_mass_cuts, events)
cut_based_sig_yield, cut_based_bg_yield = utils.getSigSidebandBGYields('bbFatJetMsd', final_mass_sig_region, events, prev_selection=cut_based_no_bb_mass_selection)

print(f"{cut_based_sig_yield = }")
print(f"{cut_based_bg_yield = }")

# Plot blinded bb fat jet mass

cut_based_mass_hist = utils.singleVarHist(events, 'bbFatJetMsd', [8, 50, 250], r"$m^{bb}$ (GeV)", selection=cut_based_no_bb_mass_selection, blind_region=final_mass_sig_region)
sig_scale = utils.getSignalPlotScaleFactor(events, selection=cut_based_no_bb_mass_selection)
plotting.ratioHistPlot(cut_based_mass_hist, bg_keys=utils.getBackgroundKeys(), sig_key=utils.getSigKey(), bg_labels=utils.getBackgroundLabels(), sig_scale=sig_scale / 3, plotdir=plotdir, name='cut_based_selection_bb_mass')


##################################################################################
# BDT analysis
##################################################################################

import xgboost as xgb

bdt_models_dir = '../bdt_models/'
model_name = 'preselection_model_sep_14'

bdtVars = [
    "MET_pt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPt",
    "VVFatJetEta",
    "VVFatJetPt",
    "VVFatJetMsd",
    "VVFatJetParticleNet_Th4q",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
]

X = np.concatenate([np.concatenate([events[key][var][:, np.newaxis] for var in bdtVars], axis=1) for key in utils.getAllKeys()], axis=0)

model = xgb.XGBClassifier()
model.load_model(f"{bdt_models_dir}/{model_name}.model")

preds = model.predict_proba(X)
