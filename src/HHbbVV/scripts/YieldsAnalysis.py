"""
Takes the skimmed parquet files (output of bbVVSkimmer),
(1) pre-selects while loading files,
(2) applies trigger efficiencies,
(3) does bb, VV jet assignment,
(4) applies basic cuts,
(5) saves data for BDT training,
(6) applies BDT cuts.

Author(s): Raghav Kansal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from os import listdir
import pickle

import utils
import plotting
from sample_labels import samples, sig_key, data_key


import importlib

_ = importlib.reload(utils)

MAIN_DIR = "../../../"
# MAIN_DIR = "./"
LUMI = {"2017": 40000}

plot_dir = f"{MAIN_DIR}/plots/ControlPlots/May9/"
os.mkdir(plot_dir)

data_dir = f"{MAIN_DIR}/../data/skimmer/Apr28/"

year = "2017"
overall_cutflow = pd.DataFrame(index=list(samples.keys()))
overall_cutflow

##################################################################################
# Load, reweight samples
##################################################################################

# Both Jet's Msds > 50 & at least one jet with Txbb > 0.8
filters = [
    [
        ("('ak8FatJetMsd', '0')", ">=", "50"),
        ("('ak8FatJetMsd', '1')", ">=", "50"),
        ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", "0.8"),
    ],
    [
        ("('ak8FatJetMsd', '0')", ">=", "50"),
        ("('ak8FatJetMsd', '1')", ">=", "50"),
        ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", "0.8"),
    ],
]

full_samples_list = listdir(f"{data_dir}/{year}")
events_dict = {}

for label, selector in samples.items():
    print(label)
    events_dict[label] = []
    for sample in full_samples_list:
        if not sample.startswith(selector):
            continue

        print(sample)

        events = pd.read_parquet(f"{data_dir}/{year}/{sample}/parquet", filters=filters)
        pickles_path = f"{data_dir}/{year}/{sample}/pickles"

        if label != data_key:
            if label == sig_key:
                n_events = utils.get_cutflow(pickles_path, year, sample)["has_4q"]
            else:
                n_events = utils.get_nevents(pickles_path, year, sample)

            events["weight"] /= n_events

        events_dict[label].append(events)

    events_dict[label] = pd.concat(events_dict[label])

utils.add_to_cutflow(events_dict, "BDTPreselection", "weight", overall_cutflow)
overall_cutflow


##################################################################################
# Apply trigger efficiencies
##################################################################################

from coffea.lookup_tools.dense_lookup import dense_lookup

with open(f"../corrections/trigEffs/AK8JetHTTriggerEfficiency_{year}.hist", "rb") as filehandler:
    ak8TrigEffs = pickle.load(filehandler)

ak8TrigEffsLookup = dense_lookup(
    np.nan_to_num(ak8TrigEffs.view(flow=False), 0), np.squeeze(ak8TrigEffs.axes.edges)
)

for sample in events_dict:
    print(sample)
    events = events_dict[sample]
    if sample == "Data":
        events["finalWeight"] = events["weight"]
    else:
        fj_trigeffs = ak8TrigEffsLookup(events["ak8FatJetPt"].values, events["ak8FatJetMsd"].values)
        # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
        combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1, keepdims=True)
        events["finalWeight"] = events["weight"] * combined_trigEffs

utils.add_to_cutflow(events_dict, "TriggerEffs", "finalWeight", overall_cutflow)

# calculate QCD scale factor
trig_yields = overall_cutflow["TriggerEffs"]
non_qcd_bgs_yield = np.sum(
    [trig_yields[sample] for sample in samples if sample not in ["HHbbVV", "QCD", "Data"]]
)
QCD_SCALE_FACTOR = (trig_yields["Data"] - non_qcd_bgs_yield) / trig_yields["QCD"]
events_dict["QCD"]["finalWeight"] *= QCD_SCALE_FACTOR

print(f"{QCD_SCALE_FACTOR = }")

utils.add_to_cutflow(events_dict, "QCD SF", "finalWeight", overall_cutflow)
overall_cutflow


##################################################################################
# bb, VV Jet Assignment
##################################################################################

bb_masks = {}

for sample, events in events_dict.items():
    txbb = events["ak8FatJetParticleNetMD_Txbb"]
    thvv = events["ak8FatJetParticleNetHWWMD_THWW4q"]
    bb_mask = txbb[0] >= txbb[1]
    bb_masks[sample] = pd.concat((bb_mask, ~bb_mask), axis=1)


##################################################################################
# Derive variables
##################################################################################

for sample, events in events_dict.items():
    print(sample)
    bb_mask = bb_masks[sample]

    fatjet_vectors = utils.make_vector(events, "ak8FatJet")
    Dijet = fatjet_vectors[:, 0] + fatjet_vectors[:, 1]

    events["DijetPt"] = Dijet.pt
    events["DijetMass"] = Dijet.M
    events["DijetEta"] = Dijet.eta

    events["bbFatJetPtOverDijetPt"] = (
        utils.get_feat(events, "bbFatJetPt", bb_mask) / events["DijetPt"]
    )
    events["VVFatJetPtOverDijetPt"] = (
        utils.get_feat(events, "VVFatJetPt", bb_mask) / events["DijetPt"]
    )
    events["VVFatJetPtOverbbFatJetPt"] = utils.get_feat(
        events, "VVFatJetPt", bb_mask
    ) / utils.get_feat(events, "bbFatJetPt", bb_mask)


##################################################################################
# Cut-based yields
##################################################################################

weight_key = "finalWeight"
mass_key = "bbFatJetMsd"

CUT_MAX_VAL = 9999

final_mass_sig_region = [100, 150]


cut_based_sel_no_bb_mass_cuts = {
    "bbFatJetPt": [250, CUT_MAX_VAL],
    "VVFatJetPt": [250, CUT_MAX_VAL],
    # 'bbFatJetMsd': [100, 150],
    "VVFatJetMsd": [110, 150],
    "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
    "VVFatJetParticleNetHWWMD_THWW4q": [0.9, CUT_MAX_VAL],
}


cut_based_no_bb_mass_selection, cut_based_no_bb_mass_cutflow = utils.make_selection(
    cut_based_sel_no_bb_mass_cuts, events_dict, bb_masks, prev_cutflow=overall_cutflow
)

cut_based_sig_yield, cut_based_bg_yield = utils.getSigSidebandBGYields(
    "bbFatJetMsd",
    final_mass_sig_region,
    events_dict,
    bb_masks,
    selection=cut_based_no_bb_mass_selection,
)

cut_based_no_bb_mass_cutflow.to_csv("cutflows/cut_based_cutflow_bdtpresel.csv")


##################################################################################
# BDT Training Data
##################################################################################

BDT_samples = [sig_key, "QCD", "TT", "Data"]

BDT_data_vars = [
    "MET_pt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPt",
    "bbFatJetMsd",
    "VVFatJetEta",
    "VVFatJetPt",
    "VVFatJetMsd",
    "VVFatJetParticleNetHWWMD_THWW4q",
    "VVFatJetParticleNetHWWMD_probQCD",
    "VVFatJetParticleNetHWWMD_probHWW3q",
    "VVFatJetParticleNetHWWMD_probHWW4q",
    "bbFatJetParticleNetMD_Txbb",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
    "finalWeight",
]

bdt_events_dict = []

for sample in BDT_samples:
    events = pd.DataFrame(
        {var: utils.get_feat(events_dict[sample], var, bb_masks[sample]) for var in BDT_data_vars}
    )
    events["Dataset"] = sample
    bdt_events_dict.append(events)

bdt_events = pd.concat(bdt_events_dict, axis=0)

import pyarrow.parquet as pq
import pyarrow as pa

table = pa.Table.from_pandas(bdt_events)
pq.write_table(table, f"{data_dir}/bdt_data.parquet")


##################################################################################
# Yields with BDT
##################################################################################

"""
Incl Txbb Threshold at 0.15 sig_eff: 0.9734
Incl Txbb Threshold at 0.2 sig_eff: 0.9602
"""

bdt_preds = np.load(f"{data_dir}/absolute_weights_preds.npy")

i = 0
for sample in BDT_samples:
    events = events_dict[sample]
    num_events = len(events)
    events["BDTScore"] = bdt_preds[i : i + num_events]
    i += num_events


bdt_cuts = {
    "BDTScore": [0.9602, CUT_MAX_VAL],
    "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
}

bdt_selection, bdt_cutflow = utils.make_selection(
    bdt_cuts, events_dict, bb_masks, prev_cutflow=overall_cutflow.drop("ST")
)

bdt_sig_yield, bdt_bg_yield = utils.getSigSidebandBGYields(
    "bbFatJetMsd",
    final_mass_sig_region,
    events_dict,
    bb_masks,
    selection=bdt_selection,
)

bdt_cutflow
bdt_sig_yield, bdt_bg_yield

bdt_cutflow.to_csv("cutflows/bdt_cutflow.csv")


post_bdt_cut_based_mass_hist = utils.singleVarHist(
    events_dict,
    "bbFatJetMsd",
    [8, 50, 250],
    r"$m^{bb}$ (GeV)",
    selection=bdt_selection,
    blind_region=final_mass_sig_region,
)

sig_scale = utils.getSignalPlotScaleFactor(events_dict, selection=bdt_selection)

plotting.ratioHistPlot(
    post_bdt_cut_based_mass_hist,
    list(events_dict.keys())[1:-1],
    sig_key,
    name="post_bdt_cuts_bb_mass",
    sig_scale=sig_scale,
)
