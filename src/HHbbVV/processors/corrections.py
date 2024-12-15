"""
Collection of utilities for corrections and systematics in processors.

Loosely based on https://github.com/jennetd/hbb-coffea/blob/master/boostedhiggs/corrections.py

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/

Authors: Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

import pickle
from pathlib import Path

import awkward as ak
import correctionlib
import hist
import numpy as np
from coffea import util as cutil
from coffea.analysis_tools import Weights
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import (
    FatJetArray,
    GenParticleArray,
    JetArray,
    MuonArray,
)

from . import utils
from .utils import P4, PAD_VAL, pad_val

ak.behavior.update(vector.behavior)
package_path = Path(__file__).parent.parent.resolve()


"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
    "jmar": ["JME", "jmar.json.gz"],
}


def get_jec_key(year: str):
    thekey = f"{year}mc"
    if year == "2016":
        thekey = "2016postVFPmc"
    elif year == "2016APV":
        thekey = "2016preVFPmc"
    return thekey


def get_vfp_year(year: str) -> str:
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"

    return year


def get_UL_year(year: str) -> str:
    return f"{get_vfp_year(year)}_UL"


def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = get_UL_year(year)
    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def get_pileup_weight(year: str, nPU: np.ndarray):
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

    return values


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray):
    """Separate wrapper function in case we just want the values separately."""
    values = get_pileup_weight(year, nPU)
    weights.add("pileup", values["nominal"], values["up"], values["down"])


kfactor_common_systs = [
    "d1K_NLO",
    "d2K_NLO",
    "d3K_NLO",
    "d1kappa_EW",
]
zsysts = kfactor_common_systs + [
    "Z_d2kappa_EW",
    "Z_d3kappa_EW",
]
znlosysts = [
    "d1kappa_EW",
    "Z_d2kappa_EW",
    "Z_d3kappa_EW",
]
wsysts = kfactor_common_systs + [
    "W_d2kappa_EW",
    "W_d3kappa_EW",
]


def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    vjets_kfactors = correctionlib.CorrectionSet.from_file(
        str(package_path / "corrections/ULvjets_corrections.json")
    )

    def get_vpt(genpart, check_offshell=False):
        """Only the leptonic samples have no resonance in the decay tree, and only
        when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
        """
        boson = ak.firsts(
            genpart[
                ((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24))
                & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
        )
        if check_offshell:
            offshell = genpart[
                genpart.hasFlags(["fromHardProcess", "isLastCopy"])
                & ak.is_none(boson)
                & (abs(genpart.pdgId) >= 11)
                & (abs(genpart.pdgId) <= 16)
            ].sum()
            return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
        return np.array(ak.fill_none(boson.pt, 0.0))

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(
                syst,
                ones,
                ewkcorr.evaluate(syst + "_up", vpt) / ewknom,
                ewkcorr.evaluate(syst + "_down", vpt) / ewknom,
            )

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


def add_ps_weight(weights, ps_weights):
    """
    Parton Shower Weights (FSR and ISR)
    "Default" variation: https://twiki.cern.ch/twiki/bin/view/CMS/HowToPDF#Which_set_of_weights_to_use
    i.e. scaling ISR up and down
    """

    nweights = len(weights.weight())
    nom = np.ones(nweights)

    up_isr = np.ones(nweights)
    down_isr = np.ones(nweights)
    up_fsr = np.ones(nweights)
    down_fsr = np.ones(nweights)

    if len(ps_weights[0]) == 4:
        up_isr = ps_weights[:, 0]  # ISR=2, FSR=1
        down_isr = ps_weights[:, 2]  # ISR=0.5, FSR=1

        up_fsr = ps_weights[:, 1]  # ISR=1, FSR=2
        down_fsr = ps_weights[:, 3]  # ISR=1, FSR=0.5

    elif len(ps_weights[0]) > 1:
        print("PS weight vector has length ", len(ps_weights[0]))

    weights.add("ISRPartonShower", nom, up_isr, down_isr)
    weights.add("FSRPartonShower", nom, up_fsr, down_fsr)

    # TODO: do we need to update sumgenweights?
    # e.g. as in https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/11e0c5225416a580d27718997a11dc3f1ec1e8d1/processor/generator.py#L74


def get_pdf_weights(events):
    """
    For the PDF acceptance uncertainty:
        - store 103 variations. 0-100 PDF values
        - The last two values: alpha_s variations.
        - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
        e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
        and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488

    Some references:
    Scale/PDF weights in MC https://twiki.cern.ch/twiki/bin/view/CMS/HowToPDF
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#PDF
    """
    return events.LHEPdfWeight.to_numpy()


def get_scale_weights(events):
    """
    QCD Scale variations, best explanation I found is here:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    TLDR: we want to vary the renormalization and factorization scales by a factor of 0.5 and 2,
    and then take the envelope of the variations on our final observation as the up/down uncertainties.

    Importantly, we need to keep track of the normalization for each variation,
    so that this uncertainty takes into account the acceptance effects of our selections.

    LHE scale variation weights (w_var / w_nominal) (from https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2018UL/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1.html#LHEScaleWeight)
    [0] is renscfact=0.5d0 facscfact=0.5d0 ; <=
    [1] is renscfact=0.5d0 facscfact=1d0 ; <=
    [2] is renscfact=0.5d0 facscfact=2d0 ;
    [3] is renscfact=1d0 facscfact=0.5d0 ; <=
    [4] is renscfact=1d0 facscfact=1d0 ;
    [5] is renscfact=1d0 facscfact=2d0 ; <=
    [6] is renscfact=2d0 facscfact=0.5d0 ;
    [7] is renscfact=2d0 facscfact=1d0 ; <=
    [8] is renscfact=2d0 facscfact=2d0 ; <=

    See also https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/11e0c5225416a580d27718997a11dc3f1ec1e8d1/processor/generator.py#L93 for an example.
    """
    if len(events[0].LHEScaleWeight) == 9:
        variations = events.LHEScaleWeight[:, [0, 1, 3, 5, 7, 8]].to_numpy()
        nominal = events.LHEScaleWeight[:, 4].to_numpy()[:, np.newaxis]
        variations /= nominal
    else:
        variations = events.LHEScaleWeight[:, [0, 1, 3, 4, 6, 7]].to_numpy()

    # clipping to avoid negative / too large weights
    return np.clip(variations, 0.0, 4.0)


def _btagSF(cset, jets, flavour, wp="M", algo="deepJet", syst="central"):
    j, nj = ak.flatten(jets), ak.num(jets)
    corrs = cset[f"{algo}_comb"] if flavour == "bc" else cset[f"{algo}_incl"]
    sf = corrs.evaluate(
        syst,
        wp,
        np.array(j.hadronFlavour),
        np.array(abs(j.eta)),
        np.array(j.pt),
    )
    return ak.unflatten(sf, nj)


def _btag_prod(eff, sf):
    num = ak.fill_none(ak.prod(1 - sf * eff, axis=-1), 1)
    den = ak.fill_none(ak.prod(1 - eff, axis=-1), 1)
    return num, den


def add_btag_weights(
    weights: Weights,
    year: str,
    jets: JetArray,
    wp: str = "M",
    algo: str = "deepJet",
):
    """Method 1b from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods"""
    get_UL_year(year)
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))
    efflookup = cutil.load(package_path / f"corrections/btag_effs/btageff_deepJet_M_{year}.coffea")

    lightJets = jets[jets.hadronFlavour == 0]
    bcJets = jets[jets.hadronFlavour > 0]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(cset, lightJets, "light", wp, algo)
    bcSF = _btagSF(cset, bcJets, "bc", wp, algo)

    lightnum, lightden = _btag_prod(lightEff, lightSF)
    bcnum, bcden = _btag_prod(bcEff, bcSF)

    weight = np.nan_to_num((1 - lightnum * bcnum) / (1 - lightden * bcden), nan=1)
    weights.add("btagSF", weight)


def add_pileupid_weights(weights: Weights, year: str, jets: JetArray, genjets, wp: str = "L"):
    """Pileup ID scale factors
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL#Data_MC_Efficiency_Scale_Factors

    Takes ak4 jets which already passed the pileup ID WP.
    Only applies to jets with pT < 50 GeV and those geometrically matched to a gen jet.
    """

    # pileup ID should only be used for jets with pT < 50
    jets = jets[jets.pt < 50]
    # check that there's a geometrically matched genjet (99.9% are, so not really necessary...)
    jets = jets[ak.any(jets.metric_table(genjets) < 0.4, axis=-1)]

    sf_cset = correctionlib.CorrectionSet.from_file(get_pog_json("jmar", year))["PUJetID_eff"]

    # save offsets to reconstruct jagged shape
    offsets = jets.pt.layout.offsets

    sfs_var = []
    for var in ["nom", "up", "down"]:
        # correctionlib < 2.3 doesn't accept jagged arrays (but >= 2.3 needs awkward v2)
        sfs = sf_cset.evaluate(ak.flatten(jets.eta), ak.flatten(jets.pt), var, wp)
        # reshape flat effs
        sfs = ak.Array(ak.layout.ListOffsetArray64(offsets, ak.layout.NumpyArray(sfs)))
        # product of SFs across arrays, automatically defaults empty lists to 1
        sfs_var.append(ak.prod(sfs, axis=1))

    weights.add("pileupID", *sfs_var)


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


# Used only for validation region right now
def add_lepton_weights(weights: Weights, year: str, lepton: MuonArray, lepton_type: str = "muon"):
    # ul_year = get_UL_year(year)

    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    lep_pt = np.array(ak.fill_none(lepton.pt, 0.0))
    lep_eta = np.abs(np.array(ak.fill_none(lepton.eta, 0.0)))

    for corr, corrDict in lepton_corrections.items():
        json_map_name = corrDict[lepton_type][year]

        # some voodoo from cristina
        lepton_pt, lepton_eta = _get_lepton_clipped(lep_pt, lep_eta, lepton_type, corr)

        values = {}
        values["nominal"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "nominal")
        values["up"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "systup")
        values["down"] = cset[json_map_name].evaluate(lepton_eta, lepton_pt, "systdown")

        # add weights (for now only the nominal weight)
        weights.add(f"{lepton_type}_{corr}", values["nominal"], values["up"], values["down"])


# For analysis region
def add_lepton_id_weights(
    weights: Weights,
    year: str,
    lepton: NanoEventsArray,
    lepton_type: str,
    wp: str,
    label: str = "",
    max_num_leptons: int = 3,
):
    year = get_vfp_year(year)
    # ul_year = get_UL_year(year)

    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    lep_exists = ak.count(lepton.pt, axis=1) > 0
    lep_pt = pad_val(lepton.pt, max_num_leptons, axis=1)
    lep_eta = pad_val(lepton.eta, max_num_leptons, axis=1)

    # some voodoo from cristina
    lepton_pt, lepton_eta = _get_lepton_clipped(lep_pt, lep_eta, lepton_type)
    values = {}

    if lepton_type == "electron":
        # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/EGM_2018_UL_electron.html
        cset_map = cset["UL-Electron-ID-SF"]

        values["nominal"] = cset_map.evaluate(year, "sf", wp, lepton_eta, lepton_pt)
        values["up"] = cset_map.evaluate(year, "sfup", wp, lepton_eta, lepton_pt)
        values["down"] = cset_map.evaluate(year, "sfdown", wp, lepton_eta, lepton_pt)
    else:
        # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/MUO_2018_UL_muon_Z_v2.html
        cset_map = cset[f"NUM_{wp}ID_DEN_TrackerMuons"]

        values["nominal"] = cset_map.evaluate(lepton_eta, lepton_pt, "nominal")
        values["up"] = cset_map.evaluate(lepton_eta, lepton_pt, "systup")
        values["down"] = cset_map.evaluate(lepton_eta, lepton_pt, "systdown")

    for key, value in values.items():
        # efficiency for a single lepton passing is 1 - (1 - eff1) * (1 - eff2) * ...
        value[lepton_pt == PAD_VAL] = 0  # if lep didn't exist, ignore efficiency
        val = 1 - np.prod(1 - value, axis=1)
        val[~lep_exists] = 1  # if no leps in event, SF = 1
        values[key] = np.nan_to_num(val, nan=1)

    # add weights (for now only the nominal weight)
    weights.add(f"{lepton_type}{label}_id_{wp}", values["nominal"], values["up"], values["down"])


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
try:
    with (package_path / "corrections/jec_compiled.pkl").open("rb") as filehandler:
        jmestuff = pickle.load(filehandler)

    ak4jet_factory = jmestuff["jet_factory"]
    fatjet_factory = jmestuff["fatjet_factory"]
except:
    print("Failed loading compiled JECs")


def _add_jec_variables(jets: JetArray, event_rho: ak.Array) -> JetArray:
    """add variables needed for JECs"""
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    # gen pT needed for smearing
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def get_jec_jets(
    events: NanoEventsArray,
    year: str,
    isData: bool = False,
    jecs: dict[str, str] = None,
    fatjets: bool = True,
) -> FatJetArray:
    """
    Based on https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py

    See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/

    If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
    """

    jec_vars = ["pt"]  # vars we're saving that are affected by JECs
    if fatjets:
        jets = events.FatJet
        jet_factory = fatjet_factory
    else:
        jets = events.Jet
        jet_factory = ak4jet_factory

    # don't apply if data
    apply_jecs = not (not ak.any(jets.pt) or isData)

    import cachetools

    jec_cache = cachetools.Cache(np.inf)

    corr_key = f"{get_vfp_year(year)}mc"

    # fatjet_factory.build gives an error if there are no fatjets in event
    if apply_jecs:
        jets = jet_factory[corr_key].build(
            _add_jec_variables(jets, events.fixedGridRhoFastjetAll), jec_cache
        )

    # return only fatjets if no jecs given
    if jecs is None:
        return jets

    jec_shifted_vars = {}

    for jec_var in jec_vars:
        tdict = {"": jets[jec_var]}
        if apply_jecs:
            for key, shift in jecs.items():
                for var in ["up", "down"]:
                    tdict[f"{key}_{var}"] = jets[shift][var][jec_var]

        jec_shifted_vars[jec_var] = tdict

    return jets, jec_shifted_vars


jmsr_vars = ["msoftdrop", "particleNet_mass"]

jmsValues = {}
jmrValues = {}

# https://github.com/cms-nanoAOD/nanoAOD-tools/blob/959c9ffb084bc974fb26ba2db41e3369cee04ae7/python/postprocessing/modules/jme/jetmetHelperRun2.py#L85-L110

# jet mass resolution: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
# nominal, down, up (these are switched in the github!!!)
jmrValues["msoftdrop"] = {
    "2016": [1.0, 0.8, 1.2],
    "2017": [1.09, 1.04, 1.14],
    # Use 2017 values for 2018 until 2018 are released
    "2018": [1.09, 1.04, 1.14],
}

# jet mass scale
# W-tagging PUPPI softdrop JMS values: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging
# 2016 values
jmsValues["msoftdrop"] = {
    "2016": [1.00, 0.9906, 1.0094],  # nominal, down, up
    "2017": [0.982, 0.978, 0.986],
    # Use 2017 values for 2018 until 2018 are released
    "2018": [0.982, 0.978, 0.986],
}

# https://github.com/cmantill/NanoNN/blob/6bd117357e2d7ec66866b5f74790e747411efcad/python/producers/hh4bProducer.py#L154-L159

# nominal, down, up
jmrValues["particleNet_mass"] = {
    "2016": [1.028, 1.007, 1.063],
    "2017": [1.026, 1.009, 1.059],
    "2018": [1.031, 1.006, 1.075],
}
jmsValues["particleNet_mass"] = {
    "2016": [1.00, 0.998, 1.002],
    "2017": [1.002, 0.996, 1.008],
    "2018": [0.994, 0.993, 1.001],
}


def get_jmsr(fatjets: FatJetArray, num_jets: int, year: str, isData: bool = False) -> dict:
    """Calculates post JMS/R masses and shifts"""
    jmsr_shifted_vars = {}

    for mkey in jmsr_vars:
        tdict = {}

        mass = utils.pad_val(fatjets[mkey], num_jets, axis=1)

        if isData:
            tdict[""] = mass
        else:
            # np.random.seed(seed)
            smearing = np.random.normal(size=mass.shape)
            # scale to JMR nom, down, up (minimum at 0)
            jmr_nom, jmr_down, jmr_up = (
                ((smearing * max(jmrValues[mkey][year][i] - 1, 0)) + 1) for i in range(3)
            )
            jms_nom, jms_down, jms_up = jmsValues[mkey][year]

            mass_jms = mass * jms_nom
            mass_jmr = mass * jmr_nom

            tdict[""] = mass_jms * jmr_nom
            tdict["JMS_down"] = mass_jmr * jms_down
            tdict["JMS_up"] = mass_jmr * jms_up
            tdict["JMR_down"] = mass_jms * jmr_down
            tdict["JMR_up"] = mass_jms * jmr_up

        jmsr_shifted_vars[mkey] = tdict

    return jmsr_shifted_vars


def add_trig_effs(weights: Weights, fatjets: FatJetArray, year: str, num_jets: int = 2):
    """Add the trigger efficiencies we measured in SingleMuon data"""
    with (package_path / f"corrections/trigEffs/{year}_combined.pkl").open("rb") as filehandler:
        combined = pickle.load(filehandler)

    # sum over TH4q bins
    effs_txbb = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

    ak8TrigEffsLookup = dense_lookup(
        np.nan_to_num(effs_txbb.view(flow=False), 0), np.squeeze(effs_txbb.axes.edges)
    )

    # TODO: confirm that these should be corrected pt, msd values
    fj_trigeffs = ak8TrigEffsLookup(
        pad_val(fatjets.Txbb, num_jets, axis=1),
        pad_val(fatjets.pt, num_jets, axis=1),
        pad_val(fatjets.msoftdrop, num_jets, axis=1),
    )

    # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
    combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1)

    weights.add("trig_effs", combined_trigEffs)


# ------------------- Lund plane reweighting ------------------- #


MAX_PT_FPARAMS = 3  # max order (+1) of pt extrapolation functions
MAX_PT_BIN = 350  # have to use subjet pt extrapolation for subjet pT > this

# caching these after loading once
(
    lp_year,
    lp_sample,
    ratio_smeared_lookups,
    ratio_lnN_smeared_lookups,
    ratio_sys_up,
    ratio_sys_down,
    ratio_dist,
    pt_extrap_lookups_dict,
    bratio,
    ratio_edges,
) = (None, None, None, None, None, None, None, None, None, None)


def _get_lund_lookups(
    year: str, seed: int = 42, lnN: bool = True, trunc_gauss: bool = False, sample: str = None
):
    import fastjet

    dR = 0.8
    fastjet.JetDefinition(fastjet.cambridge_algorithm, dR)
    fastjet.JetDefinition(fastjet.kt_algorithm, dR)
    n_LP_sf_toys = 100

    import uproot

    # initialize lund plane scale factors lookups
    f = uproot.open(package_path / f"corrections/lp_ratios/ratio_{year[:4]}.root")

    # 3D histogram: [subjet_pt, ln(0.8/Delta), ln(kT/GeV)]
    ratio_nom = f["ratio_nom"].to_numpy()
    ratio_nom_errs = f["ratio_nom"].errors()
    ratio_edges = ratio_nom[1:]
    ratio_nom = ratio_nom[0]

    ratio_sys_up = dense_lookup(f["ratio_sys_tot_up"].to_numpy()[0], ratio_edges)
    ratio_sys_down = dense_lookup(f["ratio_sys_tot_down"].to_numpy()[0], ratio_edges)
    bratio = f["h_bl_ratio"].to_numpy()[0]

    np.random.seed(seed)
    rand_noise = np.random.normal(size=[n_LP_sf_toys, *ratio_nom.shape])

    if trunc_gauss:
        # produces array of shape ``[n_sf_toys, subjet_pt bins, ln(0.8/Delta) bins, ln(kT/GeV) bins]``
        ratio_nom_smeared = ratio_nom + (ratio_nom_errs * rand_noise)
        ratio_nom_smeared = np.maximum(ratio_nom_smeared, 0)
        # save n_sf_toys lookups
        ratio_smeared_lookups = [dense_lookup(ratio_nom, ratio_edges)] + [
            dense_lookup(ratio_nom_smeared[i], ratio_edges) for i in range(n_LP_sf_toys)
        ]
    else:
        ratio_smeared_lookups = None

    if lnN:
        # revised smearing (0s -> 1s, normal -> lnN)
        zero_noms = ratio_nom == 0
        ratio_nom[zero_noms] = 1
        ratio_nom_errs[zero_noms] = 0

        zero_bs = bratio == 0
        bratio[zero_bs] = 1
        bratio = dense_lookup(bratio, ratio_edges)

        kappa = (ratio_nom + ratio_nom_errs) / ratio_nom
        ratio_nom_smeared = ratio_nom * np.power(kappa, rand_noise)
        ratio_lnN_smeared_lookups = [dense_lookup(ratio_nom, ratio_edges)] + [
            dense_lookup(ratio_nom_smeared[i], ratio_edges) for i in range(n_LP_sf_toys)
        ]
    else:
        ratio_lnN_smeared_lookups = None

    if sample is not None:
        mc_nom = f["mc_nom"].to_numpy()[0]

        with (package_path / f"corrections/lp_ratios/signals/{year}_{sample}.hist").open(
            "rb"
        ) as histf:
            sig_lp_hist = pickle.load(histf)

        sig_tot = np.sum(sig_lp_hist.values(), axis=(1, 2), keepdims=True)
        mc_tot = np.sum(mc_nom, axis=(1, 2), keepdims=True)

        # 0s -> 1 in the ratio
        mc_sig_ratio = np.nan_to_num((mc_nom / mc_tot) / (sig_lp_hist.values() / sig_tot), nan=1.0)
        mc_sig_ratio[mc_sig_ratio == 0] = 1.0
        # mc_sig_ratio = np.clip(mc_sig_ratio, 0.5, 2.0)

        ratio_dist = dense_lookup(mc_sig_ratio, ratio_edges)
    else:
        ratio_dist = None

    # ------- pT extrapolation setup: creates lookups for all the parameters and errors ------ #

    def _np_pad(arr: np.ndarray, target: int = MAX_PT_FPARAMS):
        return np.pad(arr, ((0, target - len(arr))))

    pt_extrap_lookups_dict = {"params": [], "errs": [], "sys_up_params": [], "sys_down_params": []}

    for i in range(ratio_nom.shape[1]):
        for key in pt_extrap_lookups_dict:
            pt_extrap_lookups_dict[key].append([])

        for j in range(ratio_nom.shape[2]):
            func = f["pt_extrap"][f"func_{i + 1}_{j + 1}"]
            pt_extrap_lookups_dict["params"][-1].append(
                _np_pad(func._members["fFormula"]._members["fClingParameters"])
            )
            pt_extrap_lookups_dict["errs"][-1].append(_np_pad(func._members["fParErrors"]))
            pt_extrap_lookups_dict["sys_up_params"][-1].append(
                _np_pad(
                    f["pt_extrap"][f"func_sys_tot_up_{i + 1}_{j + 1}"]
                    ._members["fFormula"]
                    ._members["fClingParameters"]
                )
            )
            pt_extrap_lookups_dict["sys_down_params"][-1].append(
                _np_pad(
                    f["pt_extrap"][f"func_sys_tot_down_{i + 1}_{j + 1}"]
                    ._members["fFormula"]
                    ._members["fClingParameters"]
                )
            )

    for key in pt_extrap_lookups_dict:
        pt_extrap_lookups_dict[key] = np.array(pt_extrap_lookups_dict[key])

    # smear parameters according to errors for pt extrap unc.
    rand_noise = np.random.normal(size=[n_LP_sf_toys, *pt_extrap_lookups_dict["params"].shape])
    smeared_pt_params = pt_extrap_lookups_dict["params"] + (
        pt_extrap_lookups_dict["errs"] * rand_noise
    )

    for key in pt_extrap_lookups_dict:
        pt_extrap_lookups_dict[key] = dense_lookup(pt_extrap_lookups_dict[key], ratio_edges[1:])

    pt_extrap_lookups_dict["smeared_params"] = [
        dense_lookup(smeared_pt_params[i], ratio_edges[1:]) for i in range(n_LP_sf_toys)
    ]

    return (
        ratio_smeared_lookups,
        ratio_lnN_smeared_lookups,
        ratio_sys_up,
        ratio_sys_down,
        ratio_dist,
        pt_extrap_lookups_dict,
        bratio,
        ratio_edges,
    )


def _get_flat_lp_vars(lds, kt_subjets_pt):
    if len(lds) != 1:
        # flatten and save offsets to unflatten afterwards
        if type(lds.layout) is ak._ext.ListOffsetArray64:
            ld_offsets = lds.kt.layout.offsets
            flat_subjet_pt = ak.flatten(kt_subjets_pt)
        elif type(lds.layout) is ak._ext.ListArray64:
            ld_offsets = lds.layout.toListOffsetArray64(False).offsets
            flat_subjet_pt = kt_subjets_pt
    else:
        # edge case of single subjet...
        ld_offsets = [0]
        flat_subjet_pt = kt_subjets_pt

    # repeat subjet pt for each lund declustering
    flat_subjet_pt = np.repeat(flat_subjet_pt, ak.count(lds.kt, axis=1)).to_numpy()
    flat_logD = np.log(0.8 / ak.flatten(lds).Delta).to_numpy()
    flat_logkt = np.log(ak.flatten(lds).kt).to_numpy()

    return ld_offsets, flat_logD, flat_logkt, flat_subjet_pt


def _get_lund_arrays(
    events: NanoEventsArray,
    jec_fatjets: FatJetArray,
    fatjet_idx: int | ak.Array,
    num_prongs: int,
    min_pt: float = 1.0,
    ca_recluster: bool = False,
):
    """
    Gets the ``num_prongs`` subjet pTs and Delta and kT per primary LP splitting of fatjets at
    ``fatjet_idx`` in each event.

    Args:
        events (NanoEventsArray): nano events
        jec_fatjets (FatJetArray): post-JEC fatjets, used to update subjet pTs.
        fatjet_idx (int | ak.Array): fatjet index
        num_prongs (int): number of prongs / subjets per jet to reweight
        min_pt (float): minimum pT for pf candidates considered for LP declustering
        ca_recluster (bool): whether to recluster subjets with CA after kT

    Returns:
        lds, kt_subjets_vec, kt_subjets_pt: lund declusterings, subjet 4-vectors, JEC-corrected subjet pTs
    """

    # jet definitions for LP SFs
    import fastjet

    # get post-JEC / pre-JEC pT ratios, to apply to subjets
    nojec_fatjets_pt = events.FatJet.pt[np.arange(len(jec_fatjets)), fatjet_idx]
    jec_correction = (jec_fatjets.pt / nojec_fatjets_pt)[:, np.newaxis]

    dR = 0.8
    cadef = fastjet.JetDefinition(fastjet.cambridge_algorithm, dR)
    ktdef = fastjet.JetDefinition(fastjet.kt_algorithm, dR)

    recluster_def = cadef if ca_recluster else ktdef

    # get pfcands of the top-matched jets
    ak8_pfcands = events.FatJetPFCands
    ak8_pfcands = ak8_pfcands[ak8_pfcands.jetIdx == fatjet_idx]
    pfcands = events.PFCands[ak8_pfcands.pFCandsIdx]

    # need to convert to such a structure for FastJet
    pfcands_vector_ptetaphi = ak.Array(
        [
            [{kin_key: cand[kin_key] for kin_key in P4} for cand in event_cands]
            for event_cands in pfcands
        ],
        with_name="PtEtaPhiMLorentzVector",
    )

    # cluster first with kT
    kt_clustering = fastjet.ClusterSequence(pfcands_vector_ptetaphi, ktdef)
    kt_subjets = kt_clustering.exclusive_jets(num_prongs)

    kt_subjets_vec = ak.zip(
        {"x": kt_subjets.px, "y": kt_subjets.py, "z": kt_subjets.pz, "t": kt_subjets.E},
        with_name="LorentzVector",
    )

    # save subjet pT * JEC scaling
    kt_subjets_pt = kt_subjets_vec.pt * jec_correction
    # get constituents
    kt_subjet_consts = kt_clustering.exclusive_jets_constituents(num_prongs)
    kt_subjet_consts = kt_subjet_consts[kt_subjet_consts.pt > min_pt]
    kt_subjet_consts = ak.flatten(kt_subjet_consts, axis=1)

    # dummy particle to pad empty subjets. SF for these subjets will be 1
    dummy_particle = ak.Array(
        [{kin_key: 0.0 for kin_key in P4}],
        with_name="PtEtaPhiMLorentzVector",
    )

    # pad empty subjets
    kt_subjet_consts = ak.fill_none(ak.pad_none(kt_subjet_consts, 1, axis=1), dummy_particle[0])

    # then re-cluster with CA
    # won't need to flatten once https://github.com/scikit-hep/fastjet/pull/145 is released
    reclustering = fastjet.ClusterSequence(kt_subjet_consts, recluster_def)
    lds = reclustering.exclusive_jets_lund_declusterings(1)

    return lds, kt_subjets_vec, kt_subjets_pt


def _get_flat_lund_arrays(events, jec_fatjet, fatjet_idx, num_prongs):
    """Wrapper for the _get_lund_arrays and _get_flat_lp_vars functions

    returns:    lds - lund declusterings,
                kt_subjets_vec - subjet 4-vectors,
                kt_subjets_pt - JEC-corrected subjet pTs,
                ld_offsets - offsets for jagged structure,
                flat_logD - flattened log(0.8/Delta),
                flat_logkt - flattened log(kT/GeV),
                flat_subjet_pt - flattened JEC-corrected subjet pTs
    """
    lds, kt_subjets_vec, kt_subjets_pt = _get_lund_arrays(
        events, jec_fatjet, fatjet_idx, num_prongs
    )

    lds_flat = ak.flatten(lds, axis=1)
    ld_offsets, flat_logD, flat_logkt, flat_subjet_pt = _get_flat_lp_vars(lds_flat, kt_subjets_pt)

    return lds, kt_subjets_vec, kt_subjets_pt, ld_offsets, flat_logD, flat_logkt, flat_subjet_pt


def _calc_lund_SFs(
    flat_logD: np.ndarray,
    flat_logkt: np.ndarray,
    flat_subjet_pt: np.ndarray,
    ld_offsets: ak.Array,
    num_prongs: int,
    ratio_lookups: list[dense_lookup],
    pt_extrap_lookups: list[dense_lookup],
    max_pt_bin: int = MAX_PT_BIN,
    max_fparams: int = MAX_PT_FPARAMS,
    CLIP: float = 5.0,
) -> np.ndarray:
    """
    Calculates scale factors for jets based on splittings in the primary Lund Plane.

    Ratio lookup tables should be binned in [subjet_pt, ln(0.8/Delta), ln(kT/GeV)].
    pT extrapolation lookup tables should be binned [ln(0.8/Delta), ln(kT/GeV)], and the values
      are the parameters for each bin's polynomial function.

    Returns nominal scale factors for each lookup table in the ``ratio_lookups`` and
      ``pt_extrap_lookups`` lists.

    Args:
        flat_logD, flat_logkt, flat_subjet_pt, ld_offsets: numpy arrays from the ``lund_arrays`` fn
        num_prongs (int): number of prongs / subjets per jet to reweight
        ratio_lookups (List[dense_lookup]): list of lookup tables of ratios to use
        pt_extrap_lookups (List[dense_lookup]): list of lookup tables of pt extrapolation function
          parameters to use

    Returns:
        np.ndarray: SF values per jet for ratio and pt extrap lookup, shape
          ``[n_jets, len(ratio_lookups) * len(pt_extrap_lookups)]``.
    """
    # get high pT subjets for extrapolation
    high_pt_sel = flat_subjet_pt > max_pt_bin
    hpt_logD = flat_logD[high_pt_sel]
    hpt_logkt = flat_logkt[high_pt_sel]
    hpt_sjpt = 1 / flat_subjet_pt[high_pt_sel]
    # store polynomial orders for pT extrapolation
    sj_pt_orders = np.array([np.power(hpt_sjpt, i) for i in range(max_fparams)]).T

    sf_vals = []
    # could be parallelised but not sure if memory / time trade-off is worth it
    for i, ratio_lookup in enumerate(ratio_lookups):
        for j, pt_extrap_lookup in enumerate(pt_extrap_lookups):
            # only recalculate if there are multiple lookup tables
            if i == 0 or len(ratio_lookups) > 1:
                ratio_vals = ratio_lookup(flat_subjet_pt, flat_logD, flat_logkt)

            # only recalculate if there are multiple pt param lookup tables
            if j == 0 or len(pt_extrap_lookups) > 1:
                params = pt_extrap_lookup(hpt_logD, hpt_logkt)
                pt_extrap_vals = np.sum(params * sj_pt_orders, axis=1)

            ratio_vals[high_pt_sel] = pt_extrap_vals

            ratio_vals = np.clip(ratio_vals, 1.0 / CLIP, CLIP)

            if len(ld_offsets) != 1:
                # recover jagged event structure
                reshaped_ratio_vals = ak.Array(
                    ak.layout.ListOffsetArray64(ld_offsets, ak.layout.NumpyArray(ratio_vals))
                )
            else:
                # edge case where only one subjet
                reshaped_ratio_vals = ratio_vals.reshape(1, -1)

            # nominal values are product of all lund plane SFs
            sf_vals.append(
                # multiply subjet SFs per jet
                np.prod(
                    # per-subjet SF
                    ak.prod(reshaped_ratio_vals, axis=1).to_numpy().reshape(-1, num_prongs),
                    axis=1,
                )
            )

    # output shape: ``[n_jets, len(ratio_lookups) x len(pt_extrap_lookups)]``
    return np.array(sf_vals).T


def get_lund_SFs(
    year: str,
    events: NanoEventsArray,
    jec_fatjets: FatJetArray,
    fatjet_idx: int | ak.Array,
    num_prongs: int,
    gen_quarks: GenParticleArray,
    weights: np.ndarray,
    sample: str = None,
    seed: int = 42,
    trunc_gauss: bool = False,
    lnN: bool = True,
    gen_bs: GenParticleArray = None,
) -> dict[str, np.ndarray]:
    """
    Calculates scale factors for jets based on splittings in the primary Lund Plane.
    Calculates random smearings for statistical uncertainties, total up/down systematic variation,
    and subjet matching and pT extrapolation systematic uncertainties.

    Args:
        events (NanoEventsArray): nano events
        jec_fatjets (FatJetArray): post-JEC fatjets, used to update subjet pTs.
        fatjet_idx (int | ak.Array): fatjet index
        num_prongs (int): number of prongs / subjets per jet to r
        gen_quarks (GenParticleArray): gen quarks
        weights (np.ndarray): event weights, for filling the LP histogram
        seed (int, optional): seed for random smearings. Defaults to 42.
        trunc_gauss (bool, optional): use truncated gaussians for smearing. Defaults to False.
        lnN (bool, optional): use log normals for smearings. Defaults to True.
        gen_bs (GenParticleArray, optional): gen b-quarks to calculate b-quark subjet uncertainties.
          Assumes only one per event! Defaults to None i.e. don't calculate any.

    Returns:
        Dict[str, np.ndarray]: dictionary with nominal weights per jet, sys variations, and (optionally) random smearings.
    """

    # global variable to not have to load + smear LP ratios each time
    global ratio_smeared_lookups, ratio_lnN_smeared_lookups, ratio_sys_up, ratio_sys_down, ratio_dist, pt_extrap_lookups_dict, bratio, ratio_edges, lp_year, lp_sample  # noqa: PLW0603

    if (
        (lnN and ratio_lnN_smeared_lookups is None)
        or (trunc_gauss and ratio_smeared_lookups is None)
        or (lp_year != year)  # redo if different year (can change to cache every year if needed)
        or (lp_sample != sample)  # redo if different sample (can change...)
    ):
        (
            ratio_smeared_lookups,
            ratio_lnN_smeared_lookups,
            ratio_sys_up,
            ratio_sys_down,
            ratio_dist,
            pt_extrap_lookups_dict,
            bratio,
            ratio_edges,
        ) = _get_lund_lookups(year, seed, lnN, trunc_gauss, sample)
        lp_year = year
        lp_sample = sample

    ratio_nominal = ratio_lnN_smeared_lookups[0] if lnN else ratio_smeared_lookups[0]

    jec_fatjet = jec_fatjets[np.arange(len(jec_fatjets)), fatjet_idx]

    # get lund plane declusterings, subjets, and flattened LP vars
    lds, kt_subjets_vec, kt_subjets_pt, ld_offsets, flat_logD, flat_logkt, flat_subjet_pt = (
        _get_flat_lund_arrays(events, jec_fatjet, fatjet_idx, num_prongs)
    )

    return lds, kt_subjets_pt

    ################################################################################################
    # ---- Fill LP histogram for signal for distortion uncertainty ---- #
    ################################################################################################

    lp_hist = hist.Hist(
        hist.axis.Variable(ratio_edges[0], name="subjet_pt", label="Subjet pT [GeV]"),
        hist.axis.Variable(ratio_edges[1], name="logD", label="ln(0.8/Delta)"),
        hist.axis.Variable(ratio_edges[2], name="logkt", label="ln(kT/GeV)"),
        storage=hist.storage.Weight(),
    )

    # repeat weights for each LP splitting
    flat_weights = np.repeat(
        np.repeat(weights, num_prongs), ak.count(ak.flatten(lds.kt, axis=1), axis=1)
    )

    lp_hist.fill(
        subjet_pt=flat_subjet_pt,
        logD=flat_logD,
        logkt=flat_logkt,
        weight=flat_weights,
    )

    ################################################################################################
    # ---- get scale factors per jet + smearings for stat unc. + syst. variations + pt extrap unc. ---- #
    ################################################################################################

    sfs = {}

    if trunc_gauss:
        sfs["lp_sf"] = _calc_lund_SFs(
            flat_logD,
            flat_logkt,
            flat_subjet_pt,
            ld_offsets,
            num_prongs,
            ratio_smeared_lookups,
            [pt_extrap_lookups_dict["params"]],
        )

    if lnN:
        sfs["lp_sf_lnN"] = _calc_lund_SFs(
            flat_logD,
            flat_logkt,
            flat_subjet_pt,
            ld_offsets,
            num_prongs,
            ratio_lnN_smeared_lookups,
            [pt_extrap_lookups_dict["params"]],
        )

    print("lp sf sys")

    sfs["lp_sf_sys_down"] = _calc_lund_SFs(
        flat_logD,
        flat_logkt,
        flat_subjet_pt,
        ld_offsets,
        num_prongs,
        [ratio_sys_down],
        [pt_extrap_lookups_dict["sys_down_params"]],
    )

    sfs["lp_sf_sys_up"] = _calc_lund_SFs(
        flat_logD,
        flat_logkt,
        flat_subjet_pt,
        ld_offsets,
        num_prongs,
        [ratio_sys_up],
        [pt_extrap_lookups_dict["sys_up_params"]],
    )

    if ratio_dist is not None:
        sfs["lp_sf_dist"] = _calc_lund_SFs(
            flat_logD,
            flat_logkt,
            flat_subjet_pt,
            ld_offsets,
            num_prongs,
            [ratio_dist],
            [pt_extrap_lookups_dict["params"]],
        )

    sfs["lp_sf_pt_extrap_vars"] = _calc_lund_SFs(
        flat_logD,
        flat_logkt,
        flat_subjet_pt,
        ld_offsets,
        num_prongs,
        [ratio_lnN_smeared_lookups[0]],
        pt_extrap_lookups_dict["smeared_params"],
    )

    ################################################################################################
    # ---- get scale factors after re-clustering with +/- one prong, for subjet matching uncs. ---- #
    ################################################################################################

    # need to save these for unclustered progns uncertainty
    np_kt_subjets_vecs = []

    for shift, nps in [("down", num_prongs - 1), ("up", num_prongs + 1)]:
        # get lund plane declusterings, subjets, and flattened LP vars
        _, np_kt_subjets_vec, _, np_ld_offsets, np_flat_logD, np_flat_logkt, np_flat_subjet_pt = (
            _get_flat_lund_arrays(events, jec_fatjet, fatjet_idx, nps)
        )

        sfs[f"lp_sf_np_{shift}"] = _calc_lund_SFs(
            np_flat_logD,
            np_flat_logkt,
            np_flat_subjet_pt,
            np_ld_offsets,
            nps,
            [ratio_lnN_smeared_lookups[0]],
            [pt_extrap_lookups_dict["params"]],
        )

        np_kt_subjets_vecs.append(np_kt_subjets_vec)

    ################################################################################################
    # ---- b-quark related uncertainties ---- #
    ################################################################################################

    if gen_bs is not None:
        assert ak.all(
            ak.count(gen_bs.pt, axis=1) == 1
        ), "b-quark uncertainties only implemented for exactly 1 b-quark per jet!"
        # find closest subjet to the b-quark
        subjet_bs_dr = ak.flatten(gen_bs).delta_r(kt_subjets_vec)
        closest_sjidx = np.argmin(subjet_bs_dr, axis=1).to_numpy()
        bsj_pts = kt_subjets_pt[np.arange(len(kt_subjets_pt)), closest_sjidx]
        # add fatjet indices to get subjet for each corresponding fatjet from the flat lds
        closest_sjidx += np.arange(len(subjet_bs_dr)) * num_prongs
        bsj_lds = ak.flatten(lds[closest_sjidx], axis=1)
        bld_offsets, bflat_logD, bflat_logkt, bflat_subjet_pt = _get_flat_lp_vars(bsj_lds, bsj_pts)

        if len(bflat_logD):
            light_lp_sfs = _calc_lund_SFs(
                bflat_logD,
                bflat_logkt,
                bflat_subjet_pt,
                bld_offsets,
                1,  # 1 prong because 1 b quark
                [ratio_nominal],
                [pt_extrap_lookups_dict["params"]],
            )

            b_lp_sfs = _calc_lund_SFs(
                bflat_logD,
                bflat_logkt,
                bflat_subjet_pt,
                bld_offsets,
                1,  # 1 prong because 1 b quark
                [bratio],
                [pt_extrap_lookups_dict["params"]],
            )

            print("light lp sfs", light_lp_sfs.shape, light_lp_sfs)
            print("b lp sfs", b_lp_sfs.shape, b_lp_sfs)

            sfs["lp_sfs_bl_ratio"] = b_lp_sfs / light_lp_sfs

            print("bl ratio", sfs["lp_sfs_bl_ratio"])
        else:
            # weird edge case where b-subjet has no splittings
            sfs["lp_sfs_bl_ratio"] = 1.0

    ################################################################################################
    # ---- subjet matching uncertainties ---- #
    ################################################################################################

    matching_dR = 0.2
    sj_matched = []
    sj_matched_idx = []

    # get dR between gen quarks and subjets
    for i in range(num_prongs):
        sj_q_dr = kt_subjets_vec.delta_r(gen_quarks[:, i])
        # is quark matched to a subjet (dR < 0.2)
        sj_matched.append(ak.min(sj_q_dr, axis=1) <= matching_dR)
        # save index of closest subjet
        sj_matched_idx.append(ak.argmin(sj_q_dr, axis=1))

    sj_matched = np.array(sj_matched).T
    sj_matched_idx = np.array(sj_matched_idx).T

    # mask quarks which aren't matched to a subjet, to avoid overcounting events
    sj_matched_idx_mask = np.copy(sj_matched_idx)
    sj_matched_idx_mask[~sj_matched] = -1

    j_q_dr = gen_quarks.delta_r(jec_fatjet)
    # events with quarks at the inside boundary of the jet
    q_boundary = (j_q_dr > 0.7) * (j_q_dr <= 0.8)
    sfs["lp_sf_inside_boundary_quarks"] = np.array(np.any(q_boundary, axis=1, keepdims=True))
    # events with quarks at the outside boundary of the jet
    q_boundary = (j_q_dr > 0.8) * (j_q_dr <= 0.9)
    sfs["lp_sf_outside_boundary_quarks"] = np.array(np.any(q_boundary, axis=1, keepdims=True))

    # events which have more than one quark matched to the same subjet
    sfs["lp_sf_double_matched_event"] = np.any(
        [np.sum(sj_matched_idx_mask == i, axis=1) > 1 for i in range(num_prongs)], axis=0
    ).astype(int)[:, np.newaxis]

    # number of quarks per event which aren't matched
    sfs["lp_sf_unmatched_quarks"] = np.sum(~sj_matched, axis=1, keepdims=True)

    # OLD pT extrapolation uncertainty
    sfs["lp_sf_num_sjpt_gt350"] = np.sum(kt_subjets_vec.pt > 350, axis=1, keepdims=True).to_numpy()

    # ------------- check unmatched quarks after +/- one prong reclustering --------------#
    unmatched_quarks = [~sj_matched]

    for np_kt_subjets_vec in np_kt_subjets_vecs:
        sj_matched = []

        # get dR between gen quarks and subjets
        for i in range(num_prongs):
            sj_q_dr = np_kt_subjets_vec.delta_r(gen_quarks[:, i])
            # is quark matched to a subjet (dR < 0.2)
            sj_matched.append(ak.min(sj_q_dr, axis=1) <= matching_dR)

        sj_matched = np.array(sj_matched).T
        unmatched_quarks.append(~sj_matched)

    # quarks which are not matched in any of the reclusterings
    unmatched_quarks = np.prod(unmatched_quarks, axis=0)
    sfs["lp_sf_rc_unmatched_quarks"] = np.sum(unmatched_quarks, axis=1, keepdims=True)

    return sfs, lp_hist
