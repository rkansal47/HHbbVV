import os
import numpy as np
import gzip
import pickle
import correctionlib
import awkward as ak
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.nanoaod import MuonArray, JetArray
from coffea.nanoevents.methods.base import NanoEventsArray
import pathlib

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "jec": ["JME", "fatJet_jerc.json.gz"],
}


def get_vfp_year(year: str):
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"

    return year


def get_UL_year(year: str):
    return f"{get_vfp_year(year)}_UL"


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


def _get_lepton_clipped(lep_pt, lep_eta, lepton_type, corr=None):
    """Some voodoo from cristina related to SF binning (needs comments!!)"""
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


def add_lepton_weights(weights: Weights, year: str, lepton: MuonArray, lepton_type: str = "muon"):
    ul_year = get_UL_year(year)

    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    lep_pt = np.array(ak.fill_none(lepton.pt, 0.0))
    lep_eta = np.abs(np.array(ak.fill_none(lepton.eta, 0.0)))

    for corr, corrDict in lepton_corrections.items():
        json_map_name = corrDict[lepton_type][year]

        # some voodoo from cristina
        lepton_pt, lepton_eta = _get_lepton_clipped(lep_pt, lep_eta, lepton_type, corr)

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


# find corrections path using this file's path
package_path = str(pathlib.Path(__file__).parent.parent.resolve())
with gzip.open(package_path + "/data/jec_compiled.pkl.gz", "rb") as filehandler:
    jmestuff = pickle.load(filehandler)

fatjet_factory = jmestuff["fatjet_factory"]


def _add_jec_variables(jets: JetArray, event_rho: ak.Array):
    """variables needed for JECs"""
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    # gen pT needed for smearing
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def get_jec_jets(events: NanoEventsArray, year: str):
    """
    Based on https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    Eventually update to V5 JECs once I figure out what's going on with the 2017 UL V5 JER scale factors
    """
    import cachetools

    jec_cache = cachetools.Cache(np.inf)

    corr_key = f"{get_vfp_year(year)}mc"

    fatjets = fatjet_factory[corr_key].build(
        _add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache
    )

    return fatjets


# giving up on doing these myself for now because there's no 2017 UL V5 JER scale factors ???
# jec_stack_names = [
#     "Summer19UL17_V5_MC_L1FastJet_AK8PFPuppi",
#     "Summer19UL17_V5_MC_L2Relative_AK8PFPuppi",
#     "Summer19UL17_V5_MC_L3Absolute_AK8PFPuppi",
#     "Summer19UL17_V5_MC_L2L3Residual_AK8PFPuppi",
# ]
