"""
Takes the skimmed parquet files (output of bbVVSkimmer),
(1) applies trigger efficiencies,
(2) does bb, VV jet assignment,
(3) makes control plots.

Author(s): Raghav Kansal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")

from PyPDF2 import PdfFileMerger
import vector

import os
from os import listdir
import pickle

import utils
import plotting
from sample_labels import samples, sig_key, data_key


MAIN_DIR = "../../../"
# MAIN_DIR = "./"
LUMI = {"2017": 40000}

plot_dir = f"{MAIN_DIR}/plots/ControlPlots/Jun27/"
os.mkdir(plot_dir)

data_dir = f"{MAIN_DIR}/../data/skimmer/Apr28/"

year = "2017"
overall_cutflow = pd.DataFrame(index=list(samples.keys()))


##################################################################################
# Load, reweight samples
##################################################################################

full_samples_list = listdir(f"{data_dir}/{year}")
events_dict = {}

for label, selector in samples.items():
    print(label)
    events_dict[label] = []
    for sample in full_samples_list:
        if not sample.startswith(selector):
            continue

        print(sample)

        events = pd.read_parquet(f"{data_dir}/{year}/{sample}/parquet")
        pickles_path = f"{data_dir}/{year}/{sample}/pickles"

        if label != data_key:
            if label == sig_key:
                n_events = utils.get_cutflow(pickles_path, year, sample)["has_4q"]
            else:
                n_events = utils.get_nevents(pickles_path, year, sample)

            events["weight"] /= n_events

        events_dict[label].append(events)

    events_dict[label] = pd.concat(events_dict[label])

utils.add_to_cutflow(events_dict, "Preselection", "weight", overall_cutflow)
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

events = events_dict[sig_key]
txbb = events["ak8FatJetParticleNetMD_Txbb"]
thvv = events["ak8FatJetParticleNetHWWMD_THWW4q"]
bb_mask = txbb[0] >= txbb[1]
vv_mask = thvv[0] >= thvv[1]
conflict = ~np.logical_xor(bb_mask, vv_mask)
print(f"{np.sum(conflict) * 100 / len(bb_mask):.1f}% conflicting jets")

bb_score = np.max(txbb[conflict], axis=1)
VV_score = np.max(thvv[conflict], axis=1)

bb_score_min = np.min(txbb[conflict], axis=1)
VV_score_min = np.min(thvv[conflict], axis=1)


plt.figure(figsize=(15, 12))
_ = plt.hist2d(
    bb_score,
    VV_score,
    [np.linspace(0.8, 1, 11), np.linspace(0.8, 1, 11)],
    weights=np.ones(len(bb_score)) * 1.0 / len(bb_mask) * 100,
)
plt.xlabel("Txbb Score")
plt.ylabel("THVV Score")
plt.title(
    "PNet scores of highest scoring jet in events \n where same jet has best Txbb and THVV score"
)
clb = plt.colorbar()
clb.set_label("% of total signal events", loc="center", rotation=270, labelpad=40.0)
plt.savefig(f"{plot_dir}/conflict_xbb_xvv_score.pdf", bbox_inches="tight")


plt.figure(figsize=(15, 12))
_ = plt.hist2d(
    bb_score_min,
    VV_score_min,
    [np.linspace(0.8, 1, 11), np.linspace(0.8, 1, 11)],
    weights=np.ones(len(bb_score)) * 1.0 / len(bb_mask) * 100,
)
plt.xlabel("Txbb Score")
plt.ylabel("THVV Score")
plt.title(
    "PNet scores of 2nd highest scoring jet in events \n where same jet has best Txbb and THVV score"
)
clb = plt.colorbar()
clb.set_label("% of total signal events", loc="center", rotation=270, labelpad=40.0)
plt.savefig(f"{plot_dir}/conflict_xbb_xvv_score_min.pdf", bbox_inches="tight")


bb_masks = {}

for sample, events in events_dict.items():
    txbb = events["ak8FatJetParticleNetMD_Txbb"]
    thvv = events["ak8FatJetParticleNetHWWMD_THWW4q"]
    bb_mask = txbb[0] >= txbb[1]
    bb_masks[sample] = pd.concat((bb_mask, ~bb_mask), axis=1)


def get_fatjet_feat(sample: str, feat: str, bb: bool):
    return events_dict[sample][feat].values[bb_masks[sample] ^ (not bb)]


##################################################################################
# Derive variables
##################################################################################

BB = True
VV = False


def make_fatjet_vector(sample: str, bb: bool):
    return vector.array(
        {
            "pt": get_fatjet_feat(sample, "ak8FatJetPt", bb),
            "phi": get_fatjet_feat(sample, "ak8FatJetPhi", bb),
            "eta": get_fatjet_feat(sample, "ak8FatJetEta", bb),
            "M": get_fatjet_feat(sample, "ak8FatJetMsd", bb),
        }
    )


for sample, events in events_dict.items():
    print(sample)
    bbFatJet = make_fatjet_vector(sample, BB)
    VVFatJet = make_fatjet_vector(sample, VV)
    Dijet = bbFatJet + VVFatJet

    events["DijetPt"] = Dijet.pt
    events["DijetMass"] = Dijet.M
    events["DijetEta"] = Dijet.eta

    events["bbFatJetPtOverDijetPt"] = get_fatjet_feat(sample, "ak8FatJetPt", BB) / events["DijetPt"]
    events["VVFatJetPtOverDijetPt"] = get_fatjet_feat(sample, "ak8FatJetPt", VV) / events["DijetPt"]
    events["VVFatJetPtOverbbFatJetPt"] = get_fatjet_feat(
        sample, "ak8FatJetPt", VV
    ) / get_fatjet_feat(sample, "ak8FatJetPt", BB)


##################################################################################
# Control plots
##################################################################################

sig_scale = np.sum(events_dict["Data"]["finalWeight"]) / np.sum(events_dict[sig_key]["finalWeight"])

sig_scale

hists = {}

# {var: (bins, label)}
hist_vars = {
    # "MET_pt": ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),
    # "DijetEta": ([50, -8, 8], r"$\eta^{jj}$"),
    # "DijetPt": ([50, 0, 750], r"$p_T^{jj}$ (GeV)"),
    # "DijetMass": ([50, 0, 2500], r"$m^{jj}$ (GeV)"),
    # "bbFatJetEta": ([50, -3, 3], r"$\eta^{bb}$"),
    # "bbFatJetPt": ([50, 200, 1000], r"$p^{bb}_T$ (GeV)"),
    # "bbFatJetMsd": ([50, 20, 250], r"$m^{bb}$ (GeV)"),
    # "bbFatJetParticleNetMD_Txbb": ([50, 0, 1], r"$p^{bb}_{Txbb}$"),
    # "VVFatJetEta": ([50, -3, 3], r"$\eta^{VV}$"),
    # "VVFatJetPt": ([50, 200, 1000], r"$p^{VV}_T$ (GeV)"),
    # "VVFatJetMsd": ([50, 20, 500], r"$m^{VV}$ (GeV)"),
    "VVFatJetParticleNet_Th4q": ([50, 0, 1], r"Probability($H\to 4q$)"),
    "VVFatJetParticleNetHWWMD_THWW4q": ([50, 0, 1], r"Probability($H\to VV\to 4q$)"),
    # "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    # "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    # "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.5], r"$p^{VV}_T / p^{bb}_T$"),
    # "BDTScore": ([50, 0, 1], r"BDT Score"),
}

for var, (bins, label) in hist_vars.items():
    if var not in hists:
        print(var)
        hists[var] = utils.singleVarHist(
            events_dict, var, bins, label, bb_masks, weight_key="finalWeight"
        )

with open(f"{plot_dir}/hists.pkl", "wb") as f:
    pickle.dump(hists, f)

merger_control_plots = PdfFileMerger()

for var in hist_vars.keys():
    name = f"{plot_dir}/{var}.pdf"
    plotting.ratioHistPlot(
        hists[var],
        list(samples.keys())[1:-1],
        name=name,
        sig_scale=sig_scale,
    )
    merger_control_plots.append(name)

merger_control_plots.write(f"{plot_dir}/ControlPlots.pdf")
merger_control_plots.close()

overall_cutflow.to_csv("cutflows/overall_cutflow.csv")
