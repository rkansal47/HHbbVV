import os
import numpy as np
import gzip
import pickle
import correctionlib
import awkward as ak
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.nanoaod import MuonArray
from coffea.nanoevents.methods.base import NanoEventsArray

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
}

def get_jec_key(year: str):
    thekey = f"{year}mc"
    if year == "2016":
        thekey = "2016postVFPmc"
    elif year == "2016APV":
        thekey = "2016preVFPmc"
    return thekey

def get_UL_year(year: str):
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"

    return f"{year}_UL"


def get_pog_json(obj: str, year: str):
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = get_UL_year(year)
    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray):
    """
    Should be able to do something similar to lepton weight but w pileup
    e.g. see here: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/LUMI_puWeights_Run2_UL/
    """
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))

    year_to_corr = {
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }

    values = {}

    values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
    values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

    # add weights (for now only the nominal weight)
    weights.add("pileup", values["nominal"], values["up"], values["down"])

def get_vpt(genpart,check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only                                                                                                                                                            
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)                                                                                                                                                                
    """
    boson = ak.firsts(genpart[
        ((genpart.pdgId == 23)|(abs(genpart.pdgId) == 24))
        & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
    ])
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11) & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.))

def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    znlosysts = [
        "d1kappa_EW",
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(syst, ones, ewkcorr.evaluate(syst + "_up", vpt) / ewknom, ewkcorr.evaluate(syst + "_down", vpt) / ewknom)

    if "ZJetsToQQ_HT" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = vjets_kfactors["ULZ_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "DYJetsToLL" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = 1
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(znlosysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset or "WJetsToLNu" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = vjets_kfactors["ULW_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)

# for scale factor validation region selection
lepton_corrections = {
    "trigger_noniso": {
        "muon": {
            "2016APV": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",  # preVBP
            "2016": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",  # postVBF
            "2017": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2018": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        },
    },
    # NOTE: We do not have SFs for mini-isolation yet
    "id": {
        "muon": {
            "2016APV": "NUM_TightID_DEN_TrackerMuons",
            "2016": "NUM_TightID_DEN_TrackerMuons",
            "2017": "NUM_TightID_DEN_TrackerMuons",
            "2018": "NUM_TightID_DEN_TrackerMuons",
        },
    },
}


def add_lepton_weights(weights: Weights, year: str, lepton: MuonArray, lepton_type: str = "muon"):
    ul_year = get_UL_year(year)
    if lepton_type == "electron":
        ul_year = ul_year.replace("_UL", "")

    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    # some voodoo from cristina related to SF binning (needs comments!!)
    def get_clip(lep_pt, lep_eta, lepton_type, corr=None):
        clip_pt = [0.0, 2000]
        clip_eta = [-2.4999, 2.4999]
        if lepton_type == "electron":
            clip_pt = [10.0, 499.999]
            if corr == "reco":
                clip_pt = [20.1, 499.999]
        elif lepton_type == "muon":
            clip_pt = [30.0, 1000.0]
            clip_eta = [0.0, 2.3999]
            if corr == "trigger_noniso":
                clip_pt = [52.0, 1000.0]
        lepton_pt = np.clip(lep_pt, clip_pt[0], clip_pt[1])
        lepton_eta = np.clip(lep_eta, clip_eta[0], clip_eta[1])
        return lepton_pt, lepton_eta

    lep_pt = np.array(ak.fill_none(lepton.pt, 0.0))
    lep_eta = np.abs(np.array(ak.fill_none(lepton.eta, 0.0)))

    for corr, corrDict in lepton_corrections.items():
        json_map_name = corrDict[lepton_type][year]

        # some voodoo from cristina
        lepton_pt, lepton_eta = get_clip(lep_pt, lep_eta, lepton_type, corr)

        values = {}
        values["nominal"] = cset[json_map_name].evaluate(ul_year, lepton_eta, lepton_pt, "sf")
        values["up"] = cset[json_map_name].evaluate(ul_year, lepton_eta, lepton_pt, "systup")
        values["down"] = cset[json_map_name].evaluate(ul_year, lepton_eta, lepton_pt, "systdown")

        # add weights (for now only the nominal weight)
        weights.add(f"{lepton_type}_{corr}", values["nominal"], values["up"], values["down"])


TOP_PDGID = 6
GEN_FLAGS = ["fromHardProcess", "isLastCopy"]


def add_top_pt_weight(weights: Weights, events: NanoEventsArray):
    """https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting"""
    # finding the two gen tops
    tops = events.GenPart[
        (abs(events.GenPart.pdgId) == TOP_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]

    # reweighting formula from https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat
    # for POWHEG+Pythia8
    tops_sf = np.exp(0.0615 - 0.0005 * tops.pt)
    # SF is geometric mean of both tops' weight
    tops_sf = np.sqrt(tops_sf[:, 0] * tops_sf[:, 1]).to_numpy()
    weights.add("top_pt", tops_sf)
