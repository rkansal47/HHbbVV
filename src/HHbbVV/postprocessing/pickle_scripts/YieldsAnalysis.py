"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import importlib

import numpy as np
import plotting
import utils

importlib.reload(utils)
importlib.reload(plotting)


plotdir = "../plots/YieldsAnalysis/Sep15/"

import os

os.system(f"mkdir -p {plotdir}")


events = utils.load_events()

for key in utils.getAllKeys():
    print(f"Preselecting {key} events")
    cut = (
        (events[key]["bbFatJetParticleNetMD_Txbb"] > 0.8)
        * (events[key]["bbFatJetMsd"] > 50)
        * (events[key]["VVFatJetMsd"] > 50)
    )
    for var in events[key]:
        events[key][var] = events[key][var][cut]

# Just for checking
for key in utils.getAllKeys():
    print(f"{key} events: {np.sum(events[key]['finalWeight']):.2f}")


CUT_MAX_VAL = 9999

final_mass_sig_region = [100, 150]

##################################################################################
# Cut-based selection (optimized for AK15 only; TODO: optimize for hybrid mode)
##################################################################################

cut_based_sel_no_bb_mass_cuts = {
    "bbFatJetPt": [250, CUT_MAX_VAL],
    "VVFatJetPt": [250, CUT_MAX_VAL],
    # 'bbFatJetMsd': [100, 150],
    "VVFatJetMsd": [110, 150],
    "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
    "VVFatJetParticleNet_Th4q": [0.9, CUT_MAX_VAL],
}

cut_based_no_bb_mass_selection, cut_based_no_bb_mass_cutflow = utils.make_selection(
    cut_based_sel_no_bb_mass_cuts, events
)
cut_based_sig_yield, cut_based_bg_yield = utils.getSigSidebandBGYields(
    "bbFatJetMsd", final_mass_sig_region, events, prev_selection=cut_based_no_bb_mass_selection
)

print(f"{cut_based_sig_yield = }")
print(f"{cut_based_bg_yield = }")

# Plot blinded bb fat jet mass

cut_based_mass_hist = utils.singleVarHist(
    events,
    "bbFatJetMsd",
    [8, 50, 250],
    r"$m^{bb}$ (GeV)",
    selection=cut_based_no_bb_mass_selection,
    blind_region=final_mass_sig_region,
)
sig_scale = utils.getSignalPlotScaleFactor(events, selection=cut_based_no_bb_mass_selection)
plotting.ratioHistPlot(
    cut_based_mass_hist,
    bg_keys=utils.getBackgroundKeys(),
    sig_key=utils.getSigKey(),
    bg_labels=utils.getBackgroundLabels(),
    sig_scale=sig_scale / 3,
    plotdir=plotdir,
    name="cut_based_selection_bb_mass",
)


##################################################################################
# BDT analysis
##################################################################################

bdt_models_dir = "../bdt_models/"
model_name = "preselection_model_sep_14"

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

X = np.concatenate(
    [
        np.concatenate([events[key][var][:, np.newaxis] for var in bdtVars], axis=1)
        for key in utils.getAllKeys()
    ],
    axis=0,
)

# model = xgb.XGBClassifier()
# model.load_model(f"{bdt_models_dir}/{model_name}.model")
#
# preds = model.predict_proba(X)

preds = np.loadtxt(f"{bdt_models_dir}/preds.txt")
preds_split = np.split(preds, np.cumsum([len(events[key]["weight"]) for key in utils.getAllKeys()]))

for i in range(len(utils.getAllKeys())):
    events[utils.getAllKeys()[i]]["BDTScore"] = preds_split[i][:, 1]


bdt_sel_no_bb_mass_cuts = {
    "BDTScore": [0.092, CUT_MAX_VAL],
    "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
}

bdt_no_bb_mass_selection, bdt_no_bb_mass_cutflow = utils.make_selection(
    bdt_sel_no_bb_mass_cuts, events
)
bdt_sig_yield, bdt_bg_yield = utils.getSigSidebandBGYields(
    "bbFatJetMsd", final_mass_sig_region, events, selection=bdt_no_bb_mass_selection
)

bdt_no_bb_mass_cutflow

print(f"{bdt_sig_yield = }")
print(f"{bdt_bg_yield = }")

# Plot blinded bb fat jet mass

bdt_mass_hist = utils.singleVarHist(
    events,
    "bbFatJetMsd",
    [8, 50, 250],
    r"$m^{bb}$ (GeV)",
    selection=bdt_no_bb_mass_selection,
    blind_region=final_mass_sig_region,
)
sig_scale = utils.getSignalPlotScaleFactor(events, selection=bdt_no_bb_mass_selection)
plotting.ratioHistPlot(
    bdt_mass_hist,
    bg_keys=utils.getBackgroundKeys(),
    sig_key=utils.getSigKey(),
    bg_labels=utils.getBackgroundLabels(),
    sig_scale=sig_scale / 2,
    plotdir=plotdir,
    name="bdt_selection_bb_mass",
)

mass_cut = {
    "bbFatJetMsd": final_mass_sig_region,
}


bdt_bb_mass_selection, bdt_bb_mass_cutflow = utils.make_selection(
    mass_cut, events, selection=bdt_no_bb_mass_selection, cutflow=bdt_no_bb_mass_cutflow
)

hists = {}

num_bins = 15

hist_vars = {  # (bins, labels)
    # "MET_pt": ([num_bins, 0, 250], r"$p^{miss}_T$ (GeV)"),
    # "DijetEta": ([num_bins, -8, 8], r"$\eta^{jj}$"),
    "DijetPt": ([num_bins, 0, 500], r"$p_T^{jj}$ (GeV)"),
    # "DijetMass": ([15, 500, 2000], r"$m^{jj}$ (GeV)"),
    # "bbFatJetEta": ([num_bins, -3, 3], r"$\eta^{bb}$"),
    # "bbFatJetPt": ([num_bins, 250, 900], r"$p^{bb}_T$ (GeV)"),
    # # "bbFatJetMsd": ([num_bins, 20, 2num_bins], r"$m^{bb}$ (GeV)"),
    # # "bbFatJetParticleNetMD_Txbb": ([num_bins, 0, 1], r"$p^{bb}_{Txbb}$"),
    # "VVFatJetEta": ([num_bins, -3, 3], r"$\eta^{VV}$"),
    # "VVFatJetPt": ([num_bins, 250, 900], r"$p^{VV}_T$ (GeV)"),
    # "VVFatJetMsd": ([15, 50, 200], r"$m^{VV}$ (GeV)"),
    # "VVFatJetParticleNet_Th4q": ([num_bins, 0.8, 1], r"$p^{VV}_{Th4q}$"),
    # "bbFatJetPtOverDijetPt": ([num_bins, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    # "VVFatJetPtOverDijetPt": ([num_bins, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    # "VVFatJetPtOverbbFatJetPt": ([num_bins, 0.4, 2], r"$p^{VV}_T / p^{bb}_T$"),
}


for var, (bins, label) in hist_vars.items():
    hists[var] = utils.singleVarHist(
        events, var, bins, label, weight_key="finalWeight", selection=bdt_bb_mass_selection
    )

for var in hist_vars:
    plotting.ratioHistPlot(
        hists[var],
        bg_keys=utils.getBackgroundKeys(),
        sig_key=utils.getSigKey(),
        bg_labels=utils.getBackgroundLabels(),
        plotdir=plotdir,
        name=f"bdt_sig_region_{var}",
        sig_scale=1e3,
    )


##################################################################################
# BDT-based cut-based analysis
##################################################################################

post_bdt_cut_based_sel_no_bb_mass_cuts = {
    "VVFatJetParticleNet_Th4q": [0.96, CUT_MAX_VAL],
    "VVFatJetMsd": [70, 160],
    "DijetMass": [700, 2000],
    "bbFatJetPt": [300, 900],
    "VVFatJetPtOverbbFatJetPt": [0.6, 2],
    "VVFatJetPt": [250, CUT_MAX_VAL],
    "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
}

(
    post_bdt_cut_based_no_bb_mass_selection,
    post_bdt_cut_based_no_bb_mass_cutflow,
) = utils.make_selection(post_bdt_cut_based_sel_no_bb_mass_cuts, events)

post_bdt_cut_based_sig_yield, post_bdt_cut_based_bg_yield = utils.getSigSidebandBGYields(
    "bbFatJetMsd", final_mass_sig_region, events, selection=post_bdt_cut_based_no_bb_mass_selection
)

post_bdt_cut_based_no_bb_mass_cutflow

print(f"{post_bdt_cut_based_sig_yield = }")
print(f"{post_bdt_cut_based_bg_yield = }")

# Plot blinded bb fat jet mass

post_bdt_cut_based_mass_hist = utils.singleVarHist(
    events,
    "bbFatJetMsd",
    [8, 50, 250],
    r"$m^{bb}$ (GeV)",
    selection=post_bdt_cut_based_no_bb_mass_selection,
    blind_region=final_mass_sig_region,
)

sig_scale = utils.getSignalPlotScaleFactor(
    events, selection=post_bdt_cut_based_no_bb_mass_selection
)

plotting.ratioHistPlot(
    post_bdt_cut_based_mass_hist,
    bg_keys=utils.getBackgroundKeys(),
    sig_key=utils.getSigKey(),
    bg_labels=utils.getBackgroundLabels(),
    sig_scale=sig_scale / 3,
    plotdir=plotdir,
    name="post_bdt_cut_based_selection_bb_mass",
)
