"""
Takes the skimmed parquet files (output of bbVVSkimmer) and evaluates the HWW Tagger.

Author(s): Raghav Kansal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")

import os
from os import listdir
import pickle

import time
import contextlib


# MAIN_DIR = "../../../"
MAIN_DIR = "./"


@contextlib.contextmanager
def timer():
    old_time = time.monotonic()
    try:
        yield
    finally:
        new_time = time.monotonic()
        print(f"Time taken: {new_time - old_time} seconds")


def get_xsecs():
    """Load cross sections json file and evaluate if necessary"""
    import json

    with open(f"{MAIN_DIR}/data/xsecs.json") as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def split_year_sample_name(sample):
    """Split sample e.g. '2017_HHToBBVVToBBQQQQ_cHHH1' into year and sample name"""
    return sample[:4], sample[5:]


def get_nevents(pickles_path, year, sample_name):
    """Adds up nevents over all pickles in ``pickles_path`` directory"""
    out_pickles = listdir(pickles_path)

    file_name = out_pickles[0]
    with open(f"{pickles_path}/{file_name}", "rb") as file:
        out_dict = pickle.load(file)
        nevents = out_dict[year][sample_name]["nevents"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with open(f"{pickles_path}/{file_name}", "rb") as file:
            out_dict = pickle.load(file)
            nevents += out_dict[year][sample_name]["nevents"]

    return nevents


def get_cutflow(pickles_path, year, sample_name):
    """Accumulates cutflow over all pickles in ``pickles_path`` directory"""
    from coffea.processor.accumulator import accumulate

    out_pickles = listdir(pickles_path)

    file_name = out_pickles[0]
    with open(f"{pickles_path}/{file_name}", "rb") as file:
        out_dict = pickle.load(file)
        cutflow = out_dict[year][sample_name]["cutflow"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with open(f"{pickles_path}/{file_name}", "rb") as file:
            out_dict = pickle.load(file)
            cutflow = accumulate([cutflow, out_dict[year][sample_name]["cutflow"]])

    return cutflow


def getParticles(particle_list, particle_type):
    """
    Finds particles in `particle_list` of type `particle_type`

    Args:
        particle_list: array of particle pdgIds
        particle_type: can be 1) string: 'b', 'V' currently, or TODO: 2) pdgID, 3) list of pdgIds
    """

    B_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24

    if particle_type == "b":
        return abs(particle_list) == B_PDGID
    elif particle_type == "V":
        return (abs(particle_list) == W_PDGID) + (abs(particle_list) == Z_PDGID)


def get_key(events: pd.DataFrame, key: str):
    return events[key].values


def make_vector(
    events: dict,
    name: str,
    mask=None,
    num_key: int = 1,
    sig: bool = True,
    new_samples: bool = False,
):
    """
    Creates Lorentz vector from input events and beginning name, assuming events contain
    {name}Pt, {name}Phi, {name}Eta, {name}Msd variables
    Optional input mask to select certain events

    Args:
        events (dict): dict of variables and corresponding numpy arrays
        name (str): object string e.g. ak8FatJet
        mask (bool array, optional): array selecting desired events
    """
    import vector

    if mask is None:
        return vector.array(
            {
                "pt": get_key(events, f"{name}Pt"),
                "phi": get_key(events, f"{name}Phi"),
                "eta": get_key(events, f"{name}Eta"),
                "M": get_key(events, f"{name}Msd")
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass"),
            }
        )
    else:
        return vector.array(
            {
                "pt": get_key(events, f"{name}Pt")[mask],
                "phi": get_key(events, f"{name}Phi")[mask],
                "eta": get_key(events, f"{name}Eta")[mask],
                "M": get_key(events, f"{name}Msd")[mask]
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass")[mask],
            }
        )


LUMI = {"2017": 40000}
SIG_KEY = "HHToBBVVToBBQQQQ"
DATA_KEY = "JetHT"

plot_dir = f"{MAIN_DIR}/plots/TaggerAnalysis/Apr26"
os.mkdir(plot_dir)

samples_dir = f"{MAIN_DIR}/../temp_data/Apr21"
xsecs = get_xsecs()

year = "2017"

# (column name, number of subcolumns)
save_columns = [
    ("weight", 1),
    ("pileupWeight", 1),
    ("genWeight", 1),
    ("ak8FatJetPt", 2),
    ("ak8FatJetMsd", 2),
    ("ak8FatJetParticleNet_Th4q", 2),
    ("ak8FatJetParticleNetHWWMD_THWW4q", 2),
]

events_dict = {}

##################################################################################
# Signal processing
##################################################################################

# GluGluToHHTobbVV_node_cHHH1

sample_name = "GluGluToHHTobbVV_node_cHHH1_pn4q"
sample_label = "HHbbVV"
events_dict[sample_label] = {}

events = pd.read_parquet(f"{samples_dir}/{year}/{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}/{sample_name}/pickles"
n_events = get_cutflow(pickles_path, year, sample_name)["has_4q"]
events["weight"] /= n_events

# get 4-vectors
vec_keys = ["ak8FatJet", "GenHiggs", "Genbb", "GenVV", "Gen4q"]
vectors = {vec_key: make_vector(events, vec_key) for vec_key in vec_keys}

is_HVV = getParticles(get_key(events, "GenHiggsChildren"), "V")
is_Hbb = getParticles(get_key(events, "GenHiggsChildren"), "b")

genHVV = vectors["GenHiggs"][is_HVV]
genHbb = vectors["GenHiggs"][is_Hbb]

dR = 1.0
masks = []
for i in range(2):
    masks.append(vectors["ak8FatJet"][:, i].deltaR(genHVV) < dR)

HVV_masks = np.transpose(np.stack(masks))

for column, num_idx in save_columns:
    if num_idx == 1:
        events_dict[sample_label][column] = np.tile(events[column].values, 2)[HVV_masks]
    else:
        events_dict[sample_label][column] = np.nan_to_num(
            events[column].values[HVV_masks], copy=True, nan=0
        )


##################################################################################
# Background processing
##################################################################################

full_samples_list = listdir(f"{samples_dir}/{year}")

# reformat into ("column name", "idx") format for reading multiindex columns
bg_column_labels = []
for key, num_columns in save_columns:
    for i in range(num_columns):
        bg_column_labels.append(f"('{key}', '{i}')")


bg_keys = ["ST", "TT", "QCD"]

for bg_key in bg_keys:
    events_dict[bg_key] = {}
    for sample in full_samples_list:
        if bg_key not in sample:
            continue
        print(sample)
        if "HH" in sample or "GluGluH" in sample:
            continue

        # get rid of weird parquet formatting
        with timer():
            events = pd.read_parquet(
                f"{samples_dir}/{year}/{sample}/parquet",
                columns=bg_column_labels,
            )

        print("read file")

        pickles_path = f"{samples_dir}/{year}/{sample}/pickles"

        with timer():
            n_events = get_nevents(pickles_path, year, sample)

        print("n events")

        with timer():
            if sample in xsecs:
                events["weight"] = (
                    xsecs[sample]
                    # * events["pileupWeight"]
                    * LUMI[year]
                    * np.sign(events["genWeight"])
                    / (n_events * np.mean(np.sign(get_key(events, "genWeight"))))
                )
            elif DATA_KEY not in sample:
                print(f"Missing xsec for {sample}")

        print("xsecs")

        for var, num_idx in save_columns:
            if num_idx == 1:
                values = np.tile(events[var].values, 2).reshape(-1)
            else:
                values = np.reshape(events[var].values, -1)

            if var in events_dict[bg_key]:
                events_dict[bg_key][var] = np.concatenate(
                    (events_dict[bg_key][var], values), axis=0
                )
            else:
                events_dict[bg_key][var] = values

for key in bg_keys:
    print(key)
    for var, num_idx in save_columns:
        if num_idx == 1:
            events_dict[key][var] = np.tile(events_dict[key][var][:, 0], 2).reshape(-1)
        # else:
        #     events_dict[key][var] = np.reshape(events_dict[key][var], -1)


np.tile(events_dict["QCD"]["weight"][:, 0], 2).reshape(-1).shape

events_dict["QCD"]["weight"].shape
events_dict["QCD"]["ak8FatJetPt"].shape

# print weighted sample yields
for sample in events_dict:
    tot_weight = np.sum(events_dict[sample]["weight"])
    print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

##############################################
# Cuts
##############################################

"""
``cuts_dict`` will be of format:
{
    sample1: {
        "cut1var1_min_max_cut1var2...": cut1,
        "cut2var2...": cut2,
        ...
    },
    sample2...
}
"""
pt_key = "Pt"
msd_key = "Msd"
var_prefix = "ak8FatJet"

cutvars_dict = {"Pt": "pt", "Msd": "msoftdrop"}

all_cuts = [
    {pt_key: [300, 1500], msd_key: [20, 320]},
    {pt_key: [400, 600], msd_key: [60, 150]},
    # {pt_key: [300, 1500], msd_key: [110, 140]},
]

var_labels = {pt_key: "pT", msd_key: "mSD"}

cuts_dict = {}
cut_labels = {}  # labels for plot titles, formatted as "var1label: [min, max] var2label..."

for sample, events in events_dict.items():
    print(sample)
    cuts_dict[sample] = {}
    for cutvars in all_cuts:
        cutstrs = []
        cutlabel = []
        cuts = []
        for cutvar, (cutmin, cutmax) in cutvars.items():
            cutstrs.append(f"{cutvars_dict[cutvar]}_{cutmin}_{cutmax}")
            cutlabel.append(f"{var_labels[cutvar]}: [{cutmin}, {cutmax}]")
            cuts.append(events[f"{var_prefix}{cutvar}"] >= cutmin)
            cuts.append(events[f"{var_prefix}{cutvar}"] < cutmax)

        cutstr = "_".join(cutstrs)
        cut = np.prod(cuts, axis=0)
        cuts_dict[sample][cutstr] = cut.astype(bool)

        if cutstr not in cut_labels:
            cut_labels[cutstr] = " ".join(cutlabel)

cuts_dict

##############################################
# Histograms
##############################################

plot_vars = {
    "th4q": {
        "title": "Non-MD Th4q",
        "score_label": "ak8FatJetParticleNet_Th4q",
        "colour": "orange",
    },
    "thvv4q": {
        "title": "MD THVV4q",
        "score_label": "ak8FatJetParticleNetHWWMD_THWW4q",
        "colour": "green",
    },
}

samples = {"qcd": "QCD", "HHbbVV": "HHbbVV", "bulkg_hflat": "BulkGToHHFlatMass"}

with open(f"{MAIN_DIR}/plots/TaggerAnalysis/Apr24/hists.pkl", "rb") as f:
    hists = pickle.load(f)


for t, pvars in plot_vars.items():
    for cutstr in cut_labels:
        plt.figure(figsize=(16, 12))
        plt.suptitle(f"HVV FatJet {pvars['title']} Scores", y=0.95)
        plt.title(cut_labels[cutstr], fontsize=20)

        for sample, colour in [("HHbbVV", "maroon"), ("qcd", "blue")]:
            plt.stairs(
                *hists[t][cutstr][sample],
                # np.linspace(0, 1, 101)[:-1],
                # hists[t][cutstr][sample][0],
                label=f"Weaver {samples[sample]}",
                linewidth=2,
                color=colour,
                # fill=True,
            )

        for sample, colour, skip in [("HHbbVV", "red", 1), ("QCD", "deepskyblue", 4)]:
            _ = plt.hist(
                events_dict[sample][pvars["score_label"]][cuts_dict[sample][cutstr]][::skip],
                histtype="step",
                bins=np.linspace(0, 1, 101),
                label=f"Coffea {sample}",
                linewidth=2,
                color=colour,
                density=True,
                weights=events_dict[sample]["weight"][cuts_dict[sample][cutstr]][::skip],
            )

        plt.ylabel("Normalized # Jets")
        plt.xlabel(f"PNet {pvars['title']} score")
        plt.legend()
        plt.savefig(
            f"{plot_dir}/{t}_hist_{cutstr}_coffea_weaver.pdf",
            bbox_inches="tight",
        )


##############################################
# ROCs
##############################################

from sklearn.metrics import roc_curve, auc

rocs = {}
sig_key = "HHbbVV"
bg_key = "QCD"
bg_skip = 4

for cutstr in cut_labels:
    rocs[cutstr] = {}

    sig_cut = cuts_dict[sig_key][cutstr]
    bg_cut = cuts_dict[bg_key][cutstr]

    y_true = np.concatenate(
        [
            np.ones(np.sum(sig_cut)),
            np.zeros(int(np.ceil(np.sum(bg_cut) / bg_skip))),
        ]
    )

    weights = np.concatenate(
        (events_dict[sig_key]["weight"][sig_cut], events_dict[bg_key]["weight"][bg_cut][::bg_skip])
    )

    for t, pvars in plot_vars.items():
        print(t)
        score_label = pvars["score_label"]
        scores = np.concatenate(
            (
                events_dict[sig_key][score_label][sig_cut],
                events_dict[bg_key][score_label][bg_cut][::bg_skip],
            )
        )
        fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
        rocs[cutstr][t] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc(fpr, tpr)}


xlim = [0, 0.6]
ylim = [1e-6, 1]

for cutstr in cut_labels:
    plt.figure(figsize=(12, 12))
    for t, pvars in plot_vars.items():
        plt.plot(
            rocs[cutstr][t]["tpr"][::10],
            rocs[cutstr][t]["fpr"][::10],
            label=f"{pvars['title']} AUC: {rocs[cutstr][t]['auc']:.2f}",
            linewidth=2,
            color=pvars["colour"],
        )
        plt.vlines(
            x=rocs[cutstr][t]["tpr"][np.searchsorted(rocs[cutstr][t]["fpr"], 0.01)],
            ymin=0,
            ymax=0.01,
            colors=pvars["colour"],
            linestyles="dashed",
        )
    plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
    plt.yscale("log")
    plt.xlabel("Signal Eff.")
    plt.ylabel("BG Eff.")
    plt.suptitle("HVV FatJet ROC Curves", y=0.95)
    plt.title(cut_labels[cutstr], fontsize=20)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.savefig(f"{plot_dir}/roccurve_{cutstr}.pdf", bbox_inches="tight")


##################################################################################
# Plots
##################################################################################

colours = {
    "lightblue": "#a6cee3",
    "darkblue": "#1f78b4",
    "red": "#e31a1c",
    "orange": "#ff7f00",
    "green": "#80ed99",
    "marigold": "#9B7EDE",
}
sig_colours = {sig_events_keys[i]: list(colours.values())[i + 2] for i in range(num_sig)}
bg_colours = [colours["green"], colours["darkblue"], colours["lightblue"]]

bg_skip = 4

_ = plt.hist(
    bg_combined_scores_dict["QCD"][:, PT_INDEX][::bg_skip],
    np.linspace(300, 1000, 101),
    weights=bg_combined_scores_dict["QCD"][:, -1][::bg_skip],
    histtype="step",
)


ptmin = 400
ptmax = 600

msdmin = 60
msdmax = 150

sig_cut = (
    (sig_pts[sample_name] >= ptmin)
    * (sig_pts[sample_name] < ptmax)
    * (sig_msds[sample_name] >= msdmin)
    * (sig_msds[sample_name] < msdmax)
)
bgcuts = {
    key: (bg_combined_scores_dict[key][:, PT_INDEX] >= ptmin)
    * (bg_combined_scores_dict[key][:, PT_INDEX] < ptmax)
    * (bg_combined_scores_dict[key][:, MSD_INDEX] >= msdmin)
    * (bg_combined_scores_dict[key][:, MSD_INDEX] < msdmax)
    for key in bg_keys
}


plot_vars = {
    "th4q": {
        "title": "Non-MD Th4q",
        "sig_scores": sig_th4q_scores,
        "bg_score_index": TH4Q_INDEX,
    },
    "thvv4q": {
        "title": "MD Thvv4q",
        "sig_scores": sig_thvv4q_scores,
        "bg_score_index": THWWMD_INDEX,
    },
}

for t, pvars in plot_vars.items():
    plt.figure(figsize=(16, 12))
    plt.title(f"HVV FatJet {pvars['title']} Scores pT:[{ptmin}, {ptmax}] msd:[{msdmin}, {msdmax}]")
    for i in range(num_sig):
        key = sig_events_keys[i]
        _ = plt.hist(
            pvars["sig_scores"][key][sig_cut],
            histtype="step",
            bins=np.linspace(0, 1, 101),
            label=sig_events_labels[i],
            linewidth=2,
            color=sig_colours[key],
            density=True,
        )
    _ = plt.hist(
        [bg_combined_scores_dict[key][:, pvars["bg_score_index"]][bgcuts[key]] for key in bg_keys],
        weights=[bg_combined_scores_dict[key][:, -1][bgcuts[key]] for key in bg_keys],
        bins=np.linspace(0, 1, 101),
        label=bg_keys,
        linewidth=2,
        color=bg_colours,
        density=True,
        stacked=True,
    )
    plt.ylabel("Normalized # Jets")
    plt.xlabel(f"PNet {pvars['title']} score")
    plt.legend()
    plt.savefig(
        f"{plot_dir}/{t}fatjetpnetscore_pt_{ptmin}_{ptmax}_msd_{msdmin}_{msdmax}.pdf",
        bbox_inches="tight",
    )

    plt.figure(figsize=(16, 12))
    plt.title(f"HVV FatJet {pvars['title']} Scores")
    for i in range(num_sig):
        key = sig_events_keys[i]
        _ = plt.hist(
            pvars["sig_scores"][key],
            histtype="step",
            bins=np.linspace(0, 1, 101),
            label=sig_events_labels[i],
            linewidth=2,
            color=sig_colours[key],
            density=True,
        )
    _ = plt.hist(
        [bg_combined_scores_dict[key][:, pvars["bg_score_index"]][::bg_skip] for key in bg_keys],
        weights=[bg_combined_scores_dict[key][:, -1][::bg_skip] for key in bg_keys],
        bins=np.linspace(0, 1, 101),
        label=bg_keys,
        linewidth=2,
        color=bg_colours,
        density=True,
        stacked=True,
    )
    plt.ylabel("Normalized # Jets")
    plt.xlabel(f"PNet {pvars['title']} score")
    plt.legend()
    plt.savefig(f"{plot_dir}/{t}fatjetpnetscore.pdf", bbox_inches="tight")


from sklearn.metrics import roc_curve, auc

fpr = {"th4q": {}, "thvv4q": {}}
tpr = {"th4q": {}, "thvv4q": {}}
thresholds = {"th4q": {}, "thvv4q": {}}

for sample_name in sig_events_keys:
    print(sample_name)
    y_true = np.concatenate(
        [
            np.ones(len(sig_weights[sample_name])),
            np.zeros(int(np.ceil(len(bg_combined_scores_dict["QCD"]) / bg_skip))),
        ]
    )
    weights = np.concatenate(
        (sig_weights[sample_name], bg_combined_scores_dict["QCD"][:, -1][::bg_skip])
    )
    scores = np.concatenate(
        (sig_th4q_scores[sample_name], bg_combined_scores_dict["QCD"][:, TH4Q_INDEX][::bg_skip])
    )

    fpr["th4q"][sample_name], tpr["th4q"][sample_name], thresholds["th4q"][sample_name] = roc_curve(
        y_true, scores, sample_weight=weights
    )

    print("thvv4q")

    scores = np.concatenate(
        (sig_thvv4q_scores[sample_name], bg_combined_scores_dict["QCD"][:, THWWMD_INDEX][::bg_skip])
    )

    (
        fpr["thvv4q"][sample_name],
        tpr["thvv4q"][sample_name],
        thresholds["thvv4q"][sample_name],
    ) = roc_curve(y_true, scores, sample_weight=weights)


thresholds["thvv4q"][sample_name][np.argmin(np.abs(fpr["thvv4q"][sample_name] - 0.01))]

xlim = [0, 0.6]
ylim = [1e-6, 1]

plt.figure(figsize=(12, 12))
for i in range(num_sig):
    key = sig_events_keys[i]
    plt.plot(
        tpr["th4q"][key][::20],
        fpr["th4q"][key][::20],
        label=f"{sig_events_labels[i]} Non-MD Th4q",
        linewidth=2,
        color="orange",
    )
    plt.vlines(
        x=tpr["th4q"][key][np.searchsorted(fpr["th4q"][key], 0.01)],
        ymin=0,
        ymax=0.01,
        colors="orange",
        linestyles="dashed",
    )
    plt.plot(
        tpr["thvv4q"][key][::5],
        fpr["thvv4q"][key][::5],
        label=f"{sig_events_labels[i]} MD THVV4q",
        linewidth=2,
        color="green",
    )
    plt.vlines(
        x=tpr["thvv4q"][key][np.searchsorted(fpr["thvv4q"][key], 0.01)],
        ymin=0,
        ymax=0.01,
        colors="green",
        linestyles="dashed",
    )
plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
plt.yscale("log")
plt.xlabel("Signal Eff.")
plt.ylabel("BG Eff.")
plt.title("HVV FatJet ROC Curves")
plt.xlim(*xlim)
plt.ylim(*ylim)
plt.legend()
plt.savefig(f"{plot_dir}/fatjetpnetroccurve.pdf", bbox_inches="tight")


# 400 < pT < 600

fpr = {"th4q": {}, "thvv4q": {}}
tpr = {"th4q": {}, "thvv4q": {}}
thresholds = {"th4q": {}, "thvv4q": {}}

for sample_name in sig_events_keys:
    print(sample_name)
    y_true = np.concatenate(
        [
            np.ones(len(sig_weights[sample_name][sig_cut])),
            np.zeros(len(bg_combined_scores_dict["QCD"][bgcuts["QCD"]])),
        ]
    )
    weights = np.concatenate(
        (sig_weights[sample_name][sig_cut], bg_combined_scores_dict["QCD"][bgcuts["QCD"]][:, -1])
    )
    scores = np.concatenate(
        (
            sig_th4q_scores[sample_name][sig_cut],
            bg_combined_scores_dict["QCD"][bgcuts["QCD"]][:, TH4Q_INDEX],
        )
    )

    fpr["th4q"][sample_name], tpr["th4q"][sample_name], thresholds["th4q"][sample_name] = roc_curve(
        y_true, scores, sample_weight=weights
    )

    print("thvv4q")

    scores = np.concatenate(
        (
            sig_thvv4q_scores[sample_name][sig_cut],
            bg_combined_scores_dict["QCD"][bgcuts["QCD"]][:, THWWMD_INDEX],
        )
    )

    (
        fpr["thvv4q"][sample_name],
        tpr["thvv4q"][sample_name],
        thresholds["thvv4q"][sample_name],
    ) = roc_curve(y_true, scores, sample_weight=weights)

    print("done")


xlim = [0, 0.6]
ylim = [1e-6, 1]

plt.figure(figsize=(12, 12))
for i in range(num_sig):
    key = sig_events_keys[i]
    plt.plot(
        tpr["th4q"][key],
        fpr["th4q"][key],
        label=f"{sig_events_labels[i]} Non-MD Th4q",
        linewidth=2,
        color="orange",
    )
    plt.vlines(
        x=tpr["th4q"][key][np.searchsorted(fpr["th4q"][key], 0.01)],
        ymin=0,
        ymax=0.01,
        colors="orange",
        linestyles="dashed",
    )
    plt.plot(
        tpr["thvv4q"][key],
        fpr["thvv4q"][key],
        label=f"{sig_events_labels[i]} MD THVV4q",
        linewidth=2,
        color="green",
    )
    plt.vlines(
        x=tpr["thvv4q"][key][np.searchsorted(fpr["thvv4q"][key], 0.01)],
        ymin=0,
        ymax=0.01,
        colors="green",
        linestyles="dashed",
    )
plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
plt.yscale("log")
plt.xlabel("Signal Eff.")
plt.ylabel("BG Eff.")
plt.title(f"HVV FatJet ROC Curves pT:[{ptmin}, {ptmax}] msd:[{msdmin}, {msdmax}]")
plt.xlim(*xlim)
plt.ylim(*ylim)
plt.legend()
plt.savefig(
    f"{plot_dir}/fatjetpnetroccurve_pt_{ptmin}_{ptmax}_msd_{msdmin}_{msdmax}.pdf",
    bbox_inches="tight",
)


##################################################################################
# Pileup Weight Check
##################################################################################


import gzip

with gzip.open("src/HHbbVV/corrections/corrections.pkl.gz", "rb") as filehandler:
    corrections = pickle.load(filehandler)

plt.scatter(corrections["2017_pileupweight"]._axes[1:], corrections["2017_pileupweight"]._values)
plt.yscale("log")
plt.xlabel("nPU")
plt.ylabel("Pileup Reweighting")

get_key(sig_events, "pileupWeight")

plt.hist(get_key(sig_events, "pileupWeight"), bins=np.logspace(1e-13, 1e-2, 11))

plt.hist(get_key(sig_events, "pileupWeight"), bins=np.linspace(1e-11, 1e-10, 101))
plt.xscale("log")
