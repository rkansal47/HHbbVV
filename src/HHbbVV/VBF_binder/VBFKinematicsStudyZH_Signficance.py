from __future__ import annotations

import os
import pickle
import warnings
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Callable

import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from coffea import nanoevents
from tqdm import tqdm

from HHbbVV.processors.utils import pad_val

warnings.filterwarnings("ignore")

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 24})

parser = ArgumentParser()
parser.add_argument("--tightID", action="store_true", help="Use tight ID")
parser.add_argument("--puID", action="store_true", help="Use pileup ID")
parser.add_argument("--min-eta-jj", action="store_true", help="Use maminx eta_jj to select jets")
parser.add_argument("--output", type=str, default="", help="Output file name")
args = parser.parse_args()

tightID = args.tightID
puID = args.puID
min_eta_jj = args.min_eta_jj
print(f"{tightID=}, {puID=}, {min_eta_jj=}")


def can_write_to_path(file_path):
    file_path = Path(file_path)
    directory = file_path.parent

    # Check if the directory exists
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return False

    # Check if the directory is writable
    if not os.access(directory, os.W_OK):
        print(f"Directory {directory} is not writable")
        return False

    # Check if the file already exists and is writable (if applicable)
    if file_path.exists() and not os.access(file_path, os.W_OK):
        print(f"File {file_path} is not writable")
        return False

    return True


if args.output is not None and args.output != "" and not can_write_to_path(args.output):
    print("Please provide a valid output path")
    exit(1)


"""Load the data"""
files = {
    "vbf": "root://cmseos.fnal.gov///store/user/lpcpfnano/cmantill/v2_3/2018/HH/VBF_HHTobbVV_CV_1_C2V_0_C3_1_TuneCP5_13TeV-madgraph-pythia8/VBF_HHTobbVV_CV_1_C2V_0_C3_1/220808_150000/0000/nano_mc2018_1-1.root",
    "qcd": "root://cmseos.fnal.gov///store/user/lpcpfnano/cmantill/v2_3/2018/QCD/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/QCD_HT1500to2000_PSWeights_madgraph/220808_163124/0000/nano_mc2018_1-1.root",
    "tt": "root://cmseos.fnal.gov///store/user/lpcpfnano/cmantill/v2_3/2018/TTbar/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic/220808_151154/0000/nano_mc2018_1-1.root",
}

events_dict = {}

for key, file in files.items():
    events_dict[key] = nanoevents.NanoEventsFactory.from_root(
        file, schemaclass=nanoevents.NanoAODSchema
    ).events()
print(f"Events loaded: {list(events_dict.keys())}")

Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25
b_PDGID = 5
GEN_FLAGS = ["fromHardProcess", "isLastCopy"]

"""Gen Quarks for VBF"""
events = events_dict["vbf"]
higgs = events.GenPart[
    (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
]

higgs_children = higgs.children

# finding bb and VV children
is_bb = abs(higgs_children.pdgId) == b_PDGID
is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)

Hbb = higgs[ak.any(is_bb, axis=2)]
HVV = higgs[ak.any(is_VV, axis=2)]

# make sure we're only getting one Higgs
Hbb = ak.pad_none(Hbb, 1, axis=1)[:, 0]
HVV = ak.pad_none(HVV, 1, axis=1)[:, 0]

vs = events.GenPart[(abs(events.GenPart.pdgId) == 24) * events.GenPart.hasFlags(GEN_FLAGS)]

# vbf output quarks are always at index 4, 5
gen_quarks = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]
print("Gen quarks calculated")

"""Baseline selections"""
num_jets = 2
sel_dict = {}
bbjet_dict = {}
vvjet_dict = {}

for key, events in events_dict.items():

    fatjets = events.FatJet

    fatjets = ak.pad_none(
        fatjets[(fatjets.pt > 300) * (fatjets.isTight) * (np.abs(fatjets.eta) <= 2.4)], 2, axis=1
    )

    # particlenet xbb vs qcd
    txbb = pad_val(
        fatjets.particleNetMD_Xbb / (fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb),
        num_jets,
        axis=1,
    )

    # bb VV assignment
    bb_mask = txbb[:, 0] >= txbb[:, 1]
    bb_mask = np.stack((bb_mask, ~bb_mask)).T

    bbjet = fatjets[bb_mask]
    vvjet = fatjets[~bb_mask]

    bbjet_dict[key] = bbjet
    vvjet_dict[key] = vvjet

    sel = ak.fill_none(
        (
            (fatjets.pt[:, 0] > 300)
            * (fatjets.pt[:, 1] > 300)
            * (np.abs(fatjets[:, 0].delta_phi(fatjets[:, 1])) > 2.6)
            * (np.abs(fatjets[:, 0].eta - fatjets[:, 1].eta) < 2.0)
        ),
        False,
    )

    if key == "vbf":
        # select only events with exactly two true VBF jets
        jets = events.Jet
        drs = jets.metric_table(gen_quarks)
        matched = ak.any(drs < 0.4, axis=2)
        two_vbf = ak.sum(matched, axis=1) == 2
        sel = sel * two_vbf

    sel_dict[key] = sel

sel_events_dict = {}
sel_bbjet_dict = {}
sel_vvjet_dict = {}

for key, sel in sel_dict.items():
    sel_events_dict[key] = events_dict[key][sel]
    sel_bbjet_dict[key] = bbjet_dict[key][sel]
    sel_vvjet_dict[key] = vvjet_dict[key][sel]

sel_gen_quarks = gen_quarks[sel_dict["vbf"]]
print("Baseline AK8 selections done.")


"""Leptons"""
electrons_dict = {}
muons_dict = {}
sel_electrons_dict = {}
sel_muons_dict = {}

for key, events in events_dict.items():
    electrons = events.Electron
    electrons = electrons[(electrons.pt > 5) & (electrons.cutBased >= electrons.LOOSE)]

    muons = events.Muon
    muons = muons[(muons.pt > 7) & (muons.looseId)]

    electrons_dict[key] = electrons
    muons_dict[key] = muons

    sel = sel_dict[key]
    sel_electrons_dict[key] = electrons[sel]
    sel_muons_dict[key] = muons[sel]


"""Matching efficiency and significance"""


def matching_efficiency(gen_quarks, vbf_jets, matching_dr=0.4, num_jets: int = 2) -> float:
    drs = ak.pad_none(vbf_jets, num_jets, axis=1)[:, :num_jets].metric_table(gen_quarks)
    matched = drs < matching_dr

    # TODO: add overlap removal?
    matching_fraction = np.mean(np.all(np.any(matched, axis=2), axis=1))
    return matching_fraction


def selection_2jets(jets):
    return jets[ak.count(jets.pt, axis=1) > 1]


def selection_2jets_etajj(jets, min_eta_jj: float = 3) -> ak.Array:
    jets = selection_2jets(jets)
    return jets[np.abs(jets[:, 0].eta - jets[:, 1].eta) > min_eta_jj]


def significance(
    sel_vbf_jets: ak.Array,
    sel_bkg_jets_dict: dict[str, ak.Array],
    selection: Callable[ak.Array, ak.Array] = lambda jets: jets[ak.count(jets.pt, axis=1) > 1],
    verbose: bool = False,
    return_effs: bool = False,  # if true, return (signfiance, signal_eff, bkg_eff)
) -> float | tuple[float, float, float]:
    total_vbf_events = len(sel_vbf_jets)
    total_vbf_selected = len(selection(sel_vbf_jets))
    signal_efficiency = total_vbf_selected / total_vbf_events
    if verbose:
        print(f"Number of total VBF events: {total_vbf_events}")
        print(f"Number of selected VBF jets: {total_vbf_selected}")
        print(f"Signal efficiency: {signal_efficiency}")

    # background efficiency
    total_bkg_events = 0
    total_bkg_selected = 0
    for key, bkg_events in sel_bkg_jets_dict.items():
        sel_bkg_jets = sel_bkg_jets_dict[key]
        k_total_bkg_events = len(bkg_events)
        k_total_bkg_selected = len(selection(sel_bkg_jets))
        total_bkg_events += k_total_bkg_events
        total_bkg_selected += k_total_bkg_selected
        if verbose:
            print(f"Number of total background events for {key}: {k_total_bkg_events}")
            print(f"Number of selected background jets for {key}: {k_total_bkg_selected}")
            print(f"Background efficiency for {key}: {k_total_bkg_selected / k_total_bkg_events}")

    background_efficiency = total_bkg_selected / total_bkg_events
    if verbose:
        print(f"Total number of background events: {total_bkg_events}")
        print(f"Total number of selected background jets: {total_bkg_selected}")
        print(f"Background efficiency: {background_efficiency}")

    # S/sqrt(B)
    # sig = signal_efficiency / np.sqrt(background_efficiency)
    sig = total_vbf_selected / np.sqrt(total_bkg_selected)

    if return_effs:
        return sig, signal_efficiency, background_efficiency
    else:
        return sig


"""Optimization over (eta_min, bbdr, vvdr, eta_jj_min, num_jets, puID, tightID)"""
if min_eta_jj:
    print("Optimizing over (eta_min, eta_max, bbdr, vvdr, puID, tightID, eta_jj_min, num_jets)...")
else:
    print("Optimizing over (eta_min, eta_max, bbdr, vvdr, puID, tightID)...")
sig_dict = {
    "pt": None,
    "etamin": None,
    "etamax": None,
    "bbdr": None,
    "vvdr": None,
    "eta_jj_min": None,
    "num_jets": None,
    "puID": None,
    "tightID": None,
    "significance": 0,
    "eff_s": 0,
    "eff_b": 0,
}
best_sel_jets = None
best_sel_bkg_jets_dict = None

pt_list = np.arange(15, 35, 5)
etamin_list = np.arange(0, 2, 0.2)
etamax_list = np.arange(4, 5.2, 0.2)
bbdr_list = np.arange(0, 2, 0.2)
vvdr_list = np.arange(0, 2, 0.2)
total_iterations = (
    len(pt_list) * len(etamin_list) * len(etamax_list) * len(bbdr_list) * len(vvdr_list)
)
eta_jj_min_list = np.arange(1, 5, 0.2)
num_jets_list = np.arange(3, 5, 1)

vars_name = ["pt", "etamin", "etamax", "bbdr", "vvdr"]
if min_eta_jj:
    vars_name.extend(["eta_jj_min", "num_jets"])
optimization_history = {
    "vars_name": vars_name,
    "vars": [],
    "significance": [],
    "eff_s": [],
    "eff_b": [],
}

for pt, etamin, etamax, bbdr, vvdr in tqdm(
    product(pt_list, etamin_list, etamax_list, bbdr_list, vvdr_list), total=total_iterations
):
    ak4_jet_selection = {
        "pt": pt,
        "eta_min": etamin,
        "eta_max": etamax,
        "dR_fatjetbb": bbdr,
        "dR_fatjetVV": vvdr,
    }

    # perform AK4 jet selection for each type
    sel_jets_dict = {}
    sel_mask_dict = {}
    for key, events in sel_events_dict.items():
        jets = events.Jet
        bbjet = sel_bbjet_dict[key]
        vvjet = sel_vvjet_dict[key]
        electrons = sel_electrons_dict[key]
        muons = sel_muons_dict[key]
        # dR_fatjetVV = 0.8 used from last two cells of VBFgenInfoTests.ipynb with data generated from SM signal vbf
        # https://github.com/rkansal47/HHbbVV/blob/vbf_systematics/src/HHbbVV/VBF_binder/VBFgenInfoTests.ipynb
        # (0-14R1R2study.parquet) has columns of different nGoodVBFJets corresponding to R1 and R2 cuts
        sel_mask = (
            # jets.isTight
            (jets.pt >= ak4_jet_selection["pt"])
            & (np.abs(jets.eta) < ak4_jet_selection["eta_max"])
            & (np.abs(jets.eta) >= ak4_jet_selection["eta_min"])
            # medium puId https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
            # & ((jets.pt > 50) | ((jets.puId & 2) == 2))
            & (bbjet.delta_r(jets) > ak4_jet_selection["dR_fatjetbb"])
            & (vvjet.delta_r(jets) > ak4_jet_selection["dR_fatjetVV"])
            & ak.all(jets.metric_table(electrons) > 0.4, axis=2)
            & ak.all(jets.metric_table(muons) > 0.4, axis=2)
        )
        if tightID:
            sel_mask = sel_mask & jets.isTight
        if puID:
            sel_mask = sel_mask & ((jets.pt > 50) | ((jets.puId & 2) == 2))

        sel_jets = jets[sel_mask]
        sel_jets_dict[key] = sel_jets
        sel_mask_dict[key] = sel_mask

    sel_bkg_jets_dict = {k: v for k, v in sel_jets_dict.items() if k != "vbf"}
    sel_vbf_jets = sel_jets_dict["vbf"]

    if not min_eta_jj:
        # calculate significance
        sel_bkg_jets_dict = {k: v for k, v in sel_jets_dict.items() if k != "vbf"}

        sig, eff_s, eff_b = significance(
            sel_vbf_jets=sel_vbf_jets,
            sel_bkg_jets_dict=sel_bkg_jets_dict,
            selection=selection_2jets_etajj,
            verbose=False,
            return_effs=True,
        )

        vars = [pt, etamin, etamax, bbdr, vvdr]
        optimization_history["vars"].append(vars)
        optimization_history["significance"].append(sig)
        optimization_history["eff_s"].append(eff_s)
        optimization_history["eff_b"].append(eff_b)

        if sig > sig_dict["significance"]:
            sig_dict["pt"] = pt
            sig_dict["etamin"] = etamin
            sig_dict["etamax"] = etamax
            sig_dict["bbdr"] = bbdr
            sig_dict["vvdr"] = vvdr
            sig_dict["puID"] = puID
            sig_dict["tightID"] = tightID
            sig_dict["significance"] = sig
            sig_dict["eff_s"] = eff_s
            sig_dict["eff_b"] = eff_b
            best_sel_jets = sel_vbf_jets
            best_sel_bkg_jets_dict = sel_bkg_jets_dict

    else:

        def top_pt_eta_min(jets, eta_jj_min=2.0, num_jets=3):
            """
            Find highest pt pair of jets with |eta_jj| > eta_jj_min.
            If no such pair is found, return the pair with the highest pt.
            """
            jets = ak.pad_none(jets, num_jets, clip=True)
            eta = jets.eta

            etas = []
            i_s = []
            # only consider the pairing among top num_jets jets
            for i in range(num_jets):
                for j in range(i + 1, num_jets):
                    etajj = ak.fill_none(np.abs(eta[:, i] - eta[:, j]) >= eta_jj_min, False)
                    etas.append(etajj)
                    i_s.append([i, j])

            inds = np.zeros((len(jets), 2))
            inds[:, 1] += 1

            eta_jj_cache = ~etas[0]
            for n in range(1, len(etas)):
                inds[eta_jj_cache * etas[n]] = i_s[n]
                eta_jj_cache = eta_jj_cache * ~etas[n]

            # select the highest pt pair of jets that satisfy eta_jj > eta_jj_min
            i1 = inds[:, 0].astype(int)
            i2 = inds[:, 1].astype(int)

            j1 = jets[np.arange(len(jets)), i1]
            j2 = jets[np.arange(len(jets)), i2]

            selected_jets = ak.concatenate([ak.unflatten(j1, 1), ak.unflatten(j2, 1)], axis=1)

            return selected_jets

        for eta_jj_min, num_jets in product(eta_jj_min_list, num_jets_list):
            # VBF
            selected_jets = top_pt_eta_min(
                sel_vbf_jets,
                eta_jj_min=eta_jj_min,
                num_jets=num_jets,
            )

            # Background
            selected_bkg_dict = {}
            for k, bkg_jets in sel_bkg_jets_dict.items():
                selected_bkgs = top_pt_eta_min(bkg_jets, eta_jj_min=etamin, num_jets=num_jets)
                selected_bkg_dict[k] = selected_bkgs

            # Calculate significance
            sig, eff_s, eff_b = significance(
                sel_vbf_jets=selected_jets,
                sel_bkg_jets_dict=selected_bkg_dict,
                selection=selection_2jets_etajj,
                verbose=False,
                return_effs=False,
            )
            vars = [pt, etamin, etamax, bbdr, vvdr, eta_jj_min, num_jets]
            optimization_history["vars"].append(vars)
            optimization_history["significance"].append(sig)
            optimization_history["eff_s"].append(eff_s)
            optimization_history["eff_b"].append(eff_b)

            # Update best significance if new significance is better
            if sig > sig_dict["significance"]:
                sig_dict["pt"] = pt
                sig_dict["etamin"] = etamin
                sig_dict["etamax"] = etamax
                sig_dict["bbdr"] = bbdr
                sig_dict["vvdr"] = vvdr
                sig_dict["puID"] = puID
                sig_dict["tightID"] = tightID
                sig_dict["eta_jj_min"] = eta_jj_min
                sig_dict["num_jets"] = num_jets
                sig_dict["significance"] = sig
                sig_dict["eff_s"] = eff_s
                sig_dict["eff_b"] = eff_b

                best_sel_jets = selected_jets
                best_sel_bkg_jets_dict = selected_bkg_dict

print("Optimization done")
print("=" * 80)

print(sig_dict)

print("=" * 80)

print("Select events with at least 2 jets")
sig, sig_eff, bkg_eff = significance(
    sel_vbf_jets=best_sel_jets,
    sel_bkg_jets_dict=best_sel_bkg_jets_dict,
    selection=selection_2jets,
    verbose=True,
    return_effs=True,
)
print(f"significance: {sig}")
print(f"True Positive Rate: {sig_eff}")
print(f"False Positive Rate: {bkg_eff}")

print("=" * 80)

print("Select events with at least 2 jets and |eta_jj| > 3")
sig, sig_eff, bkg_eff = significance(
    sel_vbf_jets=best_sel_jets,
    sel_bkg_jets_dict=best_sel_bkg_jets_dict,
    selection=selection_2jets_etajj,
    verbose=True,
    return_effs=True,
)
print(f"significance: {sig}")
print(f"True Positive Rate: {sig_eff}")
print(f"False Positive Rate: {bkg_eff}")

print("=" * 80)

# Save optimization history
if args.output is not None and args.output != "":
    output_path = Path(args.output)
    print(f"Saving optimization history to {output_path}")
    with output_path.open("wb") as f:
        pickle.dump(optimization_history, f)
