"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")

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
        particle_type: can be 1) string: 'b', 'V' or 'H' currently, or TODO: 2) pdgID, 3) list of pdgIds
    """

    B_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24

    if particle_type == "b":
        return abs(particle_list) == B_PDGID
    elif particle_type == "V":
        return (abs(particle_list) == W_PDGID) + (abs(particle_list) == Z_PDGID)


def get_key(
    events: pd.DataFrame, key: str, num_key: int = 1, sig: bool = True, new_samples: bool = False
):
    if new_samples:
        return events[key].values

    if sig:
        skip = 4 // num_key
    else:
        skip = 2 // num_key

    arr = events[key][::skip].values.reshape(-1, num_key)
    return arr


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
                "pt": get_key(events, f"{name}Pt", num_key, sig, new_samples),
                "phi": get_key(events, f"{name}Phi", num_key, sig, new_samples),
                "eta": get_key(events, f"{name}Eta", num_key, sig, new_samples),
                "M": get_key(events, f"{name}Msd", num_key, sig, new_samples)
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass", num_key, sig, new_samples),
            }
        )
    else:
        return vector.array(
            {
                "pt": get_key(events, f"{name}Pt", num_key, sig, new_samples)[mask],
                "phi": get_key(events, f"{name}Phi", num_key, sig, new_samples)[mask],
                "eta": get_key(events, f"{name}Eta", num_key, sig, new_samples)[mask],
                "M": get_key(events, f"{name}Msd", num_key, sig, new_samples)[mask]
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass", num_key, sig, new_samples)[mask],
            }
        )


LUMI = {"2017": 40000}
SIG_KEY = "HHToBBVVToBBQQQQ"
DATA_KEY = ""

plot_dir = f"{MAIN_DIR}/plots/TaggerAnalysis/Feb18"
import os

os.mkdir(plot_dir)

samples_dir = f"{MAIN_DIR}/../temp_data/211210_skimmer"
xsecs = get_xsecs()

##################################################################################
# Signal processing
##################################################################################

year = "2017"
sig_events_keys = [
    "HHToBBVVToBBQQQQ_cHHH1",
    "GluGluToHHTobbVV_node_cHHH1",
    "GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow",
    # "GluGluHToWWTo4q_M-125",
    "GluGluToHHTo4V_node_cHHH1_new",
    "GluGluToHHTo4V_node_cHHH1",
    "jhu_HHbbWW",
]
sig_events_labels = [
    "Private pre-UL HHbbVV",
    "ULv1 HHbbVV",
    "HH4W JHUGen",
    "HH4V pre-UL New PFNano",
    "HH4V pre-UL Old PFNano",
    "JHU HHbbWW",
]
sig_th4q_scores = {}
sig_thvv4q_scores = {}
sig_weights = {}
sig_events = {}

num_sig = len(sig_events_keys)

# HHToBBVVToBBQQQQ_cHHH1 (preUL private)

sample_name = sig_events_keys[0]

events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"
n_events = get_cutflow(pickles_path, year, sample_name)["has_4q"]

get_cutflow(pickles_path, year, sample_name)

events["weight"] *= (np.sign(events["genWeight"]) * LUMI[year] * xsecs[sample_name]) / (
    n_events * np.mean(np.sign(events["genWeight"]))
)

# get 4-vectors
vec_keys = ["ak8FatJet", "ak15FatJet", "GenHiggs", "Genbb", "GenVV"]
vectors = {vec_key: make_vector(events, vec_key, num_key=2, sig=True) for vec_key in vec_keys}
vectors = {**vectors, "Gen4q": make_vector(events, "Gen4q", num_key=4, sig=True)}

is_HVV = getParticles(get_key(events, "GenHiggsChildren", 2), "V")
is_Hbb = getParticles(get_key(events, "GenHiggsChildren", 2), "b")

genHVV = vectors["GenHiggs"][is_HVV]
genHbb = vectors["GenHiggs"][is_Hbb]

dR = 1.0
masks = []
for i in range(2):
    masks.append(vectors["ak15FatJet"][:, i].deltaR(genHVV) < dR)

HVV_masks = np.transpose(np.stack(masks))

sig_events[sample_name] = events
sig_th4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNet_Th4q", num_key=2)[HVV_masks], copy=True, nan=0
)
sig_thvv4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", num_key=2)[HVV_masks], copy=True, nan=0
)
sig_weights[sample_name] = np.tile(get_key(events, "weight"), [1, 2])[HVV_masks]


# GluGluToHHTobbVV_node_cHHH1

sample_name = sig_events_keys[1]

events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"
n_events = get_cutflow(pickles_path, year, sample_name)["has_4q"]
events["weight"] /= n_events

get_cutflow(pickles_path, year, sample_name)

# get 4-vectors
vec_keys = ["ak8FatJet", "ak15FatJet", "GenHiggs", "Genbb", "GenVV", "Gen4q"]
vectors = {vec_key: make_vector(events, vec_key, new_samples=True) for vec_key in vec_keys}

is_HVV = getParticles(get_key(events, "GenHiggsChildren", new_samples=True), "V")
is_Hbb = getParticles(get_key(events, "GenHiggsChildren", new_samples=True), "b")

genHVV = vectors["GenHiggs"][is_HVV]
genHbb = vectors["GenHiggs"][is_Hbb]

dR = 1.0
masks = []
for i in range(2):
    masks.append(vectors["ak15FatJet"][:, i].deltaR(genHVV) < dR)

HVV_masks = np.transpose(np.stack(masks))

sig_events[sample_name] = events
sig_th4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNet_Th4q", new_samples=True)[HVV_masks], copy=True, nan=0
)
sig_thvv4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", new_samples=True)[HVV_masks],
    copy=True,
    nan=0,
)
sig_weights[sample_name] = np.tile(get_key(events, "weight", new_samples=True), [1, 2])[HVV_masks]


# GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow

sample_name = sig_events_keys[2]

events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"
n_events = get_cutflow(pickles_path, year, sample_name)["has_2_4q"]
events["weight"] /= n_events

sig_events[sample_name] = events

sig_th4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNet_Th4q", new_samples=True).reshape(-1), copy=True, nan=0
)
sig_thvv4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", new_samples=True).reshape(-1),
    copy=True,
    nan=0,
)
sig_weights[sample_name] = np.repeat(get_key(events, "weight", new_samples=True).reshape(-1), 2)


# GluGluHToWWTo4q_M-125
#
# sample_name = sig_events_keys[3]
#
# events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
# pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"
# n_events = get_cutflow(pickles_path, year, sample_name)["all"]
# events["weight"] /= n_events
#
# sig_events[sample_name] = events
#
# sig_th4q_scores[sample_name] = np.nan_to_num(
#     get_key(events, "ak15FatJetParticleNet_Th4q", new_samples=True).reshape(-1), copy=True, nan=0
# )
# sig_thvv4q_scores[sample_name] = np.nan_to_num(
#     get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", new_samples=True).reshape(-1),
#     copy=True,
#     nan=0,
# )
# sig_weights[sample_name] = get_key(events, "weight", new_samples=True).reshape(-1)
#
# for sample_name in sig_weights:
#     print(f"Pre-selection {sample_name} yield: {np.sum(sig_weights[sample_name]):.2f}")


# GluGluToHHTo4V_node_cHHH1_new

sample_name = sig_events_keys[3]

events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"
n_events = get_cutflow(pickles_path, year, "GluGluToHHTo4V_node_cHHH1")["has_2_4q"]
events["weight"] /= n_events

sig_events[sample_name] = events

sig_th4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNet_Th4q", new_samples=True).reshape(-1), copy=True, nan=0
)
sig_thvv4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", new_samples=True).reshape(-1),
    copy=True,
    nan=0,
)
sig_weights[sample_name] = np.repeat(get_key(events, "weight", new_samples=True).reshape(-1), 2)


for sample_name in sig_weights:
    print(f"Pre-selection {sample_name} yield: {np.sum(sig_weights[sample_name]):.2f}")


# GluGluToHHTo4V_node_cHHH1_old

sample_name = sig_events_keys[4]

events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"
n_events = get_cutflow(pickles_path, year, sample_name)["has_2_4q"]
events["weight"] /= n_events

sig_events[sample_name] = events

sig_th4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNet_Th4q", new_samples=True).reshape(-1), copy=True, nan=0
)
sig_thvv4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", new_samples=True).reshape(-1),
    copy=True,
    nan=0,
)
sig_weights[sample_name] = np.repeat(get_key(events, "weight", new_samples=True).reshape(-1), 2)


# jhu_HHbbWW (ULv1 private)

sample_name = sig_events_keys[5]

events = pd.read_parquet(f"{samples_dir}/{year}_{sample_name}/parquet")
pickles_path = f"{samples_dir}/{year}_{sample_name}/pickles"

get_cutflow(pickles_path, year, sample_name)

n_events = get_cutflow(pickles_path, year, sample_name)["all"]

events["weight"] *= (np.sign(events["genWeight"]) * LUMI[year]) / (
    n_events * np.mean(np.sign(events["genWeight"]))
)

# get 4-vectors
vec_keys = ["ak8FatJet", "ak15FatJet", "GenHiggs", "Genbb", "GenVV"]
vectors = {vec_key: make_vector(events, vec_key, num_key=2, sig=True) for vec_key in vec_keys}
vectors = {**vectors, "Gen4q": make_vector(events, "Gen4q", num_key=4, sig=True)}

is_HVV = getParticles(get_key(events, "GenHiggsChildren", 2), "V")
is_Hbb = getParticles(get_key(events, "GenHiggsChildren", 2), "b")

genHVV = vectors["GenHiggs"][is_HVV]
genHbb = vectors["GenHiggs"][is_Hbb]

dR = 1.0
masks = []
for i in range(2):
    masks.append(vectors["ak15FatJet"][:, i].deltaR(genHVV) < dR)

HVV_masks = np.transpose(np.stack(masks))

sig_events[sample_name] = events
sig_th4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNet_Th4q", num_key=2)[HVV_masks], copy=True, nan=0
)
sig_thvv4q_scores[sample_name] = np.nan_to_num(
    get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", num_key=2)[HVV_masks], copy=True, nan=0
)
sig_weights[sample_name] = np.tile(get_key(events, "weight"), [1, 2])[HVV_masks]


for sample_name in sig_weights:
    print(f"Pre-selection {sample_name} yield: {np.sum(sig_weights[sample_name]):.2f}")


##################################################################################
# Background processing
##################################################################################

full_samples_list = listdir(samples_dir)
full_samples_list

bg_columns = [
    "weight",
    "pileupWeight",
    "genWeight",
    "ak15FatJetParticleNet_Th4q",
    "ak15FatJetParticleNetHWWMD_THWW4q",
]

bg_scores_dict = {}

for sample in full_samples_list:
    if "HH" in sample:
        continue

    year, sample_name = split_year_sample_name(sample)
    print(sample)

    # get rid of weird parquet formatting
    with timer():
        events = pd.read_parquet(
            f"{samples_dir}/{sample}/parquet",
            columns=bg_columns,
        )

    print("read file")

    pickles_path = f"{samples_dir}/{sample}/pickles"

    with timer():
        n_events = get_nevents(f"{samples_dir}/{sample}/pickles", year, sample_name)

    print("n events")

    with timer():
        if sample_name in xsecs:
            events["weight"] *= (
                xsecs[sample_name]
                # * events["pileupWeight"]
                * LUMI[year]
                * np.sign(events["genWeight"])
                / (n_events * np.mean(np.sign(get_key(events, "genWeight", 1, sig=False))))
            )
        elif DATA_KEY not in sample_name:
            print(f"Missing xsec for {sample_name}")

    print("xsecs")

    bg_scores_dict[sample] = np.concatenate(
        (
            get_key(events, "ak15FatJetParticleNet_Th4q", 2, False).reshape(-1, 1),
            get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", 2, False).reshape(-1, 1),
            np.repeat(get_key(events, "weight", 1, False), 2).reshape(-1, 1),
        ),
        axis=1,
    )

for sample in bg_scores_dict:
    tot_weight = np.sum(bg_scores_dict[sample][:, 2])
    print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

bg_scores = np.concatenate(list(bg_scores_dict.values()), axis=0)
bg_scores_dict = {}
_ = np.nan_to_num(bg_scores, False, 0)


##################################################################################
# Plots
##################################################################################

colours = {
    "lightblue": "#a6cee3",
    "darkblue": "#1f78b4",
    "red": "#e31a1c",
    "orange": "#ff7f00",
    "green": "#80ed99",
    "marigold": "#F3A712",
}
sig_colours = {sig_events_keys[i]: list(colours.values())[i + 1] for i in range(num_sig)}
bg_colour = colours["lightblue"]

bg_skip = 4

plt.figure(figsize=(16, 12))
plt.title("HVV FatJet Non-MD Th4q Scores")
for i in range(num_sig):
    key = sig_events_keys[i]
    _ = plt.hist(
        sig_th4q_scores[key],
        histtype="step",
        bins=np.linspace(0, 1, 101),
        label=sig_events_labels[i],
        linewidth=2,
        color=sig_colours[key],
        density=True,
    )
_ = plt.hist(
    bg_scores[:, 0][::bg_skip],
    weights=bg_scores[:, 2][::bg_skip],
    bins=np.linspace(0, 1, 101),
    label="Background",
    linewidth=2,
    color=bg_colour,
    density=True,
)
plt.ylabel("# Events")
plt.xlabel("PNet Non-MD TH4q score")
plt.legend()


plt.savefig(f"{plot_dir}/th4qfatjetpnetscore.pdf", bbox_inches="tight")


plt.figure(figsize=(16, 12))
plt.title("HVV FatJet MD Thvv4q Scores")
for i in range(num_sig):
    key = sig_events_keys[i]
    _ = plt.hist(
        sig_thvv4q_scores[key],
        histtype="step",
        bins=np.linspace(0, 1, 101),
        label=sig_events_labels[i],
        linewidth=2,
        color=sig_colours[key],
        density=True,
    )
_ = plt.hist(
    bg_scores[:, 1][::bg_skip],
    weights=bg_scores[:, 2][::bg_skip],
    bins=np.linspace(0, 1, 101),
    label="Background",
    linewidth=2,
    color=bg_colour,
    density=True,
)
plt.ylabel("# Events")
plt.xlabel("PNet MD THvv4q score")
plt.legend()


plt.savefig(f"{plot_dir}/thvv4qmdfatjetpnetscore.pdf", bbox_inches="tight")


def rocCurve(fpr, tpr, title=None, xlim=[0, 0.4], ylim=[1e-6, 1e-2], plotdir="", name=""):
    """Plots a ROC curve"""
    plt.figure(figsize=(12, 12))
    plt.plot(tpr, fpr)
    plt.yscale("log")
    plt.xlabel("Signal Eff.")
    plt.ylabel("BG Eff.")
    plt.title(title)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")


from sklearn.metrics import roc_curve, auc

fpr = {"th4q": {}, "thvv4q": {}}
tpr = {"th4q": {}, "thvv4q": {}}
thresholds = {"th4q": {}, "thvv4q": {}}
aucs = {"th4q": {}, "thvv4q": {}}

for sample_name in sig_events_keys[3:]:
    y_true = np.concatenate([np.ones(len(sig_weights[sample_name])), np.zeros(len(bg_scores) // 4)])
    weights = np.concatenate((sig_weights[sample_name], bg_scores[:, 2][::bg_skip]))
    scores = np.concatenate((sig_th4q_scores[sample_name], bg_scores[:, 0][::bg_skip]))

    fpr["th4q"][sample_name], tpr["th4q"][sample_name], thresholds["th4q"][sample_name] = roc_curve(
        y_true, scores, sample_weight=weights
    )

    scores = np.concatenate((sig_thvv4q_scores[sample_name], bg_scores[:, 1][::bg_skip]))

    (
        fpr["thvv4q"][sample_name],
        tpr["thvv4q"][sample_name],
        thresholds["thvv4q"][sample_name],
    ) = roc_curve(y_true, scores, sample_weight=weights)


fpr

xlim = [0, 0.6]
ylim = [1e-6, 1]

plt.figure(figsize=(12, 12))
for i in range(num_sig):
    key = sig_events_keys[i]
    plt.plot(
        tpr["th4q"][key][::20],
        fpr["th4q"][key][::20],
        label=sig_events_labels[i],
        linewidth=2,
        color=sig_colours[key],
    )
    plt.vlines(
        x=tpr["th4q"][key][np.searchsorted(fpr["th4q"][key], 0.01)],
        ymin=0,
        ymax=0.01,
        colors=sig_colours[key],
        linestyles="dashed",
    )
plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
plt.yscale("log")
plt.xlabel("Signal Eff.")
plt.ylabel("BG Eff.")
plt.title("HVV FatJet Non-MD Th4q ROC Curves")
plt.xlim(*xlim)
plt.ylim(*ylim)
plt.legend()
plt.savefig(f"{plot_dir}/th4qfatjetpnetroccurve.pdf", bbox_inches="tight")


plt.figure(figsize=(12, 12))
for i in range(num_sig):
    key = sig_events_keys[i]
    plt.plot(
        tpr["thvv4q"][key][::5],
        fpr["thvv4q"][key][::5],
        label=sig_events_labels[i],
        linewidth=2,
        color=sig_colours[key],
    )
    plt.vlines(
        x=tpr["thvv4q"][key][np.searchsorted(fpr["thvv4q"][key], 0.01)],
        ymin=0,
        ymax=0.01,
        colors=sig_colours[key],
        linestyles="dashed",
    )
plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
plt.yscale("log")
plt.xlabel("Signal Eff.")
plt.ylabel("BG Eff.")
plt.title("HVV FatJet MD Thvv4q ROC Curves")
plt.xlim(*xlim)
plt.ylim(*ylim)
plt.legend()
plt.savefig(f"{plot_dir}/thvv4qfatjetpnetroccurve.pdf", bbox_inches="tight")


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
