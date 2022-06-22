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
import pickle

import utils
import plotting
from sample_labels import samples, sig_key, data_key
from utils import CUT_MAX_VAL

import importlib

_ = importlib.reload(utils)
_ = importlib.reload(plotting)

MAIN_DIR = "../../../"
# MAIN_DIR = "./"

plot_dir = f"{MAIN_DIR}/plots/YieldsAnalysis/May26/"
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


events_dict = utils.load_samples(data_dir, samples, year, filters)

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
# Load BDT Preds
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


##################################################################################
# Scan cuts
##################################################################################

from tqdm import tqdm

MIN_BDT_CUT = 0.97
MIN_TXBB_CUT = 0.97

tight_events_dict = {}
tight_bb_masks = {}

del events_dict["ST"]

for sample in BDT_samples:
    events = events_dict[sample]
    sel = (utils.get_feat(events, "BDTScore") > MIN_BDT_CUT) * (
        utils.get_feat(events, "bbFatJetParticleNetMD_Txbb", bb_masks[sample]) > MIN_TXBB_CUT
    )
    tight_events_dict[sample] = events[sel]
    tight_bb_masks[sample] = bb_masks[sample][sel]


# overall_cutflow = overall_cutflow.drop("ScanPreselection")
utils.add_to_cutflow(tight_events_dict, "ScanPreselection", "finalWeight", overall_cutflow)
overall_cutflow

bdt_cuts = np.linspace(MIN_BDT_CUT, 1, 30, endpoint=False)
txbb_cuts = np.linspace(MIN_BDT_CUT, 1, 30, endpoint=False)

signs = []

importlib.reload(utils)

for bdt_cut in tqdm(bdt_cuts):
    bsigns = []
    for txbb_cut in txbb_cuts:
        cuts = {
            "BDTScore": [bdt_cut, CUT_MAX_VAL],
            "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
        }

        bdt_selection, bdt_cutflow = utils.make_selection(
            cuts, tight_events_dict, tight_bb_masks, prev_cutflow=overall_cutflow
        )

        bdt_sig_yield, bdt_bg_yield = utils.getSigSidebandBGYields(
            "bbFatJetMsd",
            final_mass_sig_region,
            tight_events_dict,
            tight_bb_masks,
            selection=bdt_selection,
        )

        bsigns.append(bdt_sig_yield / np.sqrt(bdt_bg_yield))

    signs.append(bsigns)


signs = np.nan_to_num(signs, nan=0, posinf=0, neginf=0)
np.max(signs)

plt.figure(figsize=(15, 12))
plt.imshow(signs[::-1], extent=[txbb_cuts[0], 1, bdt_cuts[0], 1], vmin=0.012, vmax=0.016)
plt.ylabel("BDT Cut")
plt.xlabel("Txbb Cut")
plt.colorbar()
plt.title("2017 Exp. Significances")
# plt.savefig(f"{plot_dir}/bdt_txbb_sign_scan_097_097.pdf", bbox_inches="tight")


bdt_txbb_cuts = {
    "BDTScore": [0.985, CUT_MAX_VAL],
    "bbFatJetParticleNetMD_Txbb": [0.977, CUT_MAX_VAL],
}

bdt_selection, bdt_cutflow = utils.make_selection(
    bdt_txbb_cuts, events_dict, bb_masks, prev_cutflow=overall_cutflow
)

bdt_sig_yield, bdt_bg_yield = utils.getSigSidebandBGYields(
    "bbFatJetMsd",
    final_mass_sig_region,
    events_dict,
    bb_masks,
    selection=bdt_selection,
)

bdt_sig_yield, bdt_bg_yield
bdt_sig_yield / np.sqrt(bdt_bg_yield)

# bdt_cutflow.to_csv("cutflows/_cutflow.csv")

post_bdt_cut_based_mass_hist = utils.singleVarHist(
    events_dict,
    "bbFatJetMsd",
    [20, 50, 250],
    r"$m^{bb}$ (GeV)",
    bb_masks,
    selection=bdt_selection,
    blind_region=final_mass_sig_region,
)

sig_scale = utils.getSignalPlotScaleFactor(events_dict, selection=bdt_selection)

plotting.ratioHistPlot(
    post_bdt_cut_based_mass_hist,
    list(events_dict.keys())[1:-1],
    name=f"{plot_dir}/post_bdt_985_txbb_977_cuts_bb_mass.pdf",
    sig_scale=sig_scale / 2,
)
