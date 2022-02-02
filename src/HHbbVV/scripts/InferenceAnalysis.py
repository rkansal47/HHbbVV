"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from coffea.processor import column_accumulator
import pickle

import vector


import time
import contextlib


@contextlib.contextmanager
def timer():
    old_time = time.monotonic()
    try:
        yield
    finally:
        new_time = time.monotonic()
        print(f"Time taken: {new_time - old_time} seconds")


LUMI = {"2017": 40000}
SIG_KEY = "HHToBBVVToBBQQQQ"
DATA_KEY = ""


def get_xsecs():
    """Load cross sections json file and evaluate if necessary"""
    import json

    with open("data/xsecs.json") as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def split_year_sample_name(sample):
    """Split sample e.g. '2017_HHToBBVVToBBQQQQ_cHHH1' into year and sample name"""
    return sample[:4], sample[5:]


plot_dir = "plots/TaggerAnalysis/"

# import os
# os.mkdir(plot_dir)

samples_dir = "../temp_data/211210_skimmer"
full_samples_list = listdir(samples_dir)

full_samples_list

samples_list = [
    "2017_QCD_HT1000to1500",
    "2017_HHToBBVVToBBQQQQ_cHHH1",
]

columns = ["weight", "ak15FatJetParticleNet_Th4q", "ak15FatJetParticleNetHWWMD_THWW4q"]

xsecs = get_xsecs()

# sample = samples_list[0]

full_samples_dict = {}


def get_nevents(pickles_path, year, sample_name):
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


for sample in full_samples_list:
    year, sample_name = split_year_sample_name(sample)
    print(sample)

    # get rid of weird parquet formatting
    with timer():
        events = (
            pd.read_parquet(
                f"{samples_dir}/{sample}/parquet",
                columns=None if SIG_KEY in sample_name else columns,
            )
            # .reset_index()
            # .filter(regex="(?<!entry)$")
        )

    print("read file")

    with timer():
        n_events = get_nevents(f"{samples_dir}/{sample}/pickles", year, sample_name)

    print("n events")

    with timer():
        if sample_name in xsecs:
            events["weight"] *= xsecs[sample_name] * LUMI[year] / n_events

            # temp reweighting because of weird parquet structure
            if SIG_KEY in sample_name:
                events["weight"] /= 4
            else:
                events["weight"] /= 2
        elif DATA_KEY not in sample_name:
            print(f"Missing xsec for {sample_name}")

    print("xsecs")

    # gets rid of th
    full_samples_dict[sample] = events


events = full_samples_dict["2017_HHToBBVVToBBQQQQ_cHHH1"]


def get_key(events: pd.DataFrame, key: str, num_key: int = 1, sig: bool = True):
    skip = 4 // num_key if sig else 2 // num_key
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


# get 4-vectors
vec_keys = ["ak8FatJet", "ak15FatJet", "GenHiggs", "Genbb", "GenVV", "Gen4q"]
vectors = {
    vec_key: make_vector(events, vec_key, num_key=2 if not vec_key == "Gen4q" else 4, sig=True)
    for vec_key in vec_keys
}

is_HVV = getParticles(get_key(events, "GenHiggsChildren", 2), "V")
is_Hbb = getParticles(get_key(events, "GenHiggsChildren", 2), "b")

genHVV = vectors["GenHiggs"][is_HVV]
genHbb = vectors["GenHiggs"][is_Hbb]

dR = 1.0
masks = []
for i in range(2):
    masks.append(vectors["ak15FatJet"][:, i].deltaR(genHVV) < dR)

HVV_masks = np.transpose(np.stack(masks))

sig_old_score = get_key(events, "ak15FatJetParticleNet_Th4q", 2)[HVV_masks]
sig_score = get_key(events, "ak15FatJetParticleNetHWWMD_THWW4q", 2)[HVV_masks]


plt.title("HVV FatJet PNet Scores")
_ = plt.hist(sig_old_score, histtype="step", bins=np.linspace(0, 1, 101), label="Non-MD H4q")
_ = plt.hist(sig_score, histtype="step", bins=np.linspace(0, 1, 101), label="New MD HWW4q")
plt.ylabel("# Events")
plt.xlabel("PNet Score")
plt.legend()
plt.savefig(f"{plot_dir}/hvvfatjetpnetscore.pdf")

plt.hist(sig_score, histtype="step", bins=np.linspace(0, 1, 101))
plt.hist(sig_score)
