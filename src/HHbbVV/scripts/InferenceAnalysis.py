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


def get_key(events: pd.DataFrame, key: str, num_key: int = 1, sig: bool = True):
    if sig:
        return events[key].values
    else:
        skip = 2 // num_key
        arr = events[key][::skip].values.reshape(-1, num_key)
        return arr


def make_vector(events: dict, name: str, mask=None, num_key: int = 1, sig: bool = True):
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
                "pt": get_key(events, f"{name}Pt", num_key, sig),
                "phi": get_key(events, f"{name}Phi", num_key, sig),
                "eta": get_key(events, f"{name}Eta", num_key, sig),
                "M": get_key(events, f"{name}Msd", num_key, sig)
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass", num_key, sig),
            }
        )
    else:
        return vector.array(
            {
                "pt": get_key(events, f"{name}Pt", num_key, sig)[mask],
                "phi": get_key(events, f"{name}Phi", num_key, sig)[mask],
                "eta": get_key(events, f"{name}Eta", num_key, sig)[mask],
                "M": get_key(events, f"{name}Msd", num_key, sig)[mask]
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass", num_key, sig)[mask],
            }
        )


LUMI = {"2017": 40000}
SIG_KEY = "HHToBBVVToBBQQQQ"
DATA_KEY = ""

plot_dir = f"{MAIN_DIR}/plots/TaggerAnalysis/"
samples_dir = f"{MAIN_DIR}/../temp_data/211210_skimmer"
full_samples_list = listdir(samples_dir)
xsecs = get_xsecs()

full_samples_list

##################################################################################
# Signal processing
##################################################################################

sample = "2017_GluGluToHHTobbVV_node_cHHH1"
year, sample_name = split_year_sample_name(sample)

# sig_events = pd.read_parquet(
#     f"{MAIN_DIR}/../data/2017_UL_nano/GluGluToHHTobbVV_node_cHHH1/0-1.parquet"
# )

sig_events = pd.read_parquet(f"{MAIN_DIR}/../temp_data/220208_skimmer/{sample}/parquet")

pickles_path = f"{MAIN_DIR}/../temp_data/220208_skimmer/{sample}/pickles"
n_events = get_cutflow(pickles_path, year, sample_name)["has_4q"]

sig_events["weight"] /= n_events
np.sum(sig_events["weight"])

# get 4-vectors
vec_keys = ["ak8FatJet", "ak15FatJet", "GenHiggs", "Genbb", "GenVV", "Gen4q"]
vectors = {vec_key: make_vector(sig_events, vec_key, sig=True) for vec_key in vec_keys}

is_HVV = getParticles(get_key(sig_events, "GenHiggsChildren", 2), "V")
is_Hbb = getParticles(get_key(sig_events, "GenHiggsChildren", 2), "b")

genHVV = vectors["GenHiggs"][is_HVV]
genHbb = vectors["GenHiggs"][is_Hbb]

dR = 1.0
masks = []
for i in range(2):
    masks.append(vectors["ak15FatJet"][:, i].deltaR(genHVV) < dR)

HVV_masks = np.transpose(np.stack(masks))

sig_events

sig_old_score = get_key(sig_events, "ak15FatJetParticleNet_Th4q")[HVV_masks]
sig_score = get_key(sig_events, "ak15FatJetParticleNetHWWMD_THWW4q")[HVV_masks]
sig_weight = np.tile(get_key(sig_events, "weight"), [1, 2])[HVV_masks]

_ = np.nan_to_num(sig_old_score, False, 0)

plt.hist(sig_score, np.linspace(0, 1, 101), histtype="step")

##################################################################################
# Background processing
##################################################################################

bg_columns = [
    "weight",
    "pileupWeight",
    "genWeight",
    "ak15FatJetParticleNet_Th4q",
    "ak15FatJetParticleNetHWWMD_THWW4q",
]

bg_scores_dict = {}

for sample in full_samples_list:
    year, sample_name = split_year_sample_name(sample)
    print(sample)

    sig_sample = SIG_KEY in sample_name

    # get rid of weird parquet formatting
    with timer():
        events = pd.read_parquet(
            f"{samples_dir}/{sample}/parquet",
            columns=None if sig_sample else bg_columns,
        )

    print("read file")

    pickles_path = f"{samples_dir}/{sample}/pickles"

    with timer():

        n_events = (
            # only bbVV events with bb, VV, and 4 gen quarks are counted
            get_cutflow(pickles_path, year, sample_name)["has_4q"]
            if sig_sample
            else get_nevents(f"{samples_dir}/{sample}/pickles", year, sample_name)
        )

    print("n events")

    with timer():
        if sample_name in xsecs:
            events["weight"] *= (
                xsecs[sample_name]
                # * events["pileupWeight"]
                * LUMI[year]
                * np.sign(events["genWeight"])
                / (n_events * np.mean(np.sign(get_key(events, "genWeight", 1, sig_sample))))
            )
        elif DATA_KEY not in sample_name:
            print(f"Missing xsec for {sample_name}")

    print("xsecs")

    if sig_sample:
        sig_events_preUL = events
    else:
        bg_scores_dict[sample] = np.concatenate(
            (
                get_key(events, "ak15FatJetParticleNet_Th4q", 2, False).reshape(-1, 1),
                get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", 2, False).reshape(-1, 1),
                np.repeat(get_key(events, "weight", 1, False), 2).reshape(-1, 1),
            ),
            axis=1,
        )

    # break
    # if not sig_sample:
    #     break

tot_weight = np.sum(get_key(sig_events, "weight"))
print(f"Pre-selection {SIG_KEY} yield: {tot_weight}")
for sample in bg_scores_dict:
    tot_weight = np.sum(bg_scores_dict[sample][:, 2])
    print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

bg_scores = np.concatenate(list(bg_scores_dict.values()), axis=0)
bg_scores_dict = {}
_ = np.nan_to_num(bg_scores, False, 0)


##################################################################################
# Plots
##################################################################################

plt.figure(figsize=(16, 12))
plt.title("HVV FatJet PNet Scores on HVV jets from ULv1 HHbbVV samples")
_ = plt.hist(
    sig_old_score, histtype="step", bins=np.linspace(0, 1, 101), label="Non-MD H4q", linewidth=2
)
_ = plt.hist(
    sig_score, histtype="step", bins=np.linspace(0, 1, 101), label="New MD HWW4q", linewidth=2
)
plt.ylabel("# Events")
plt.xlabel("PNet score on signal")
plt.legend()
plt.savefig(f"{plot_dir}/hvvfatjetpnetscore_ulv1.pdf", bbox_inches="tight")


plt.figure(figsize=(16, 12))
plt.title("BG FatJet PNet Scores")
_ = plt.hist(
    bg_scores[:, 0],
    histtype="step",
    bins=np.linspace(0, 1, 101),
    weights=bg_scores[:, 2],
    label="Non-MD H4q",
    linewidth=2,
)
_ = plt.hist(
    bg_scores[:, 1],
    histtype="step",
    bins=np.linspace(0, 1, 101),
    weights=bg_scores[:, 2],
    label="New MD HWW4q",
    linewidth=2,
)
plt.ylabel("# Events")
plt.xlabel("PNet score on background")
plt.legend()
plt.savefig(f"{plot_dir}/bgfatjetpnetscore.pdf", bbox_inches="tight")


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


from sklearn.metrics import roc_curve

y_true = np.concatenate([np.ones(len(sig_weight)), np.zeros(len(bg_scores))])
weights = np.concatenate((sig_weight, bg_scores[:, 2]))

print("ROC curving")

old_fpr, old_tpr, old_thresholds = roc_curve(
    y_true, np.concatenate((sig_old_score, bg_scores[:, 0])), sample_weight=weights
)

print("plotting old")
rocCurve(
    old_fpr,
    old_tpr,
    title="Old PNet Non-MD H4q Tagger",
    plotdir=plot_dir,
    name="h4q_roc",
    xlim=[0, 1],
    ylim=[1e-6, 1],
)

print("new")
fpr, tpr, thresholds = roc_curve(
    y_true, np.concatenate((sig_score, bg_scores[:, 1])), sample_weight=weights
)

print("plotting new")
rocCurve(
    fpr,
    tpr,
    title="PNet MD HVV4q Tagger",
    plotdir=plot_dir,
    name="hvv4q_roc",
    xlim=[0, 1],
    ylim=[1e-6, 1],
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
