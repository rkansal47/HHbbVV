import os
from typing import Dict, List, Tuple
import numpy as np
import gzip
import pickle
import correctionlib
import awkward as ak

from coffea.analysis_tools import Weights
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods.nanoaod import MuonArray, JetArray, FatJetArray, GenParticleArray
from coffea.nanoevents.methods.base import NanoEventsArray

from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)

import pathlib

import fastjet

from .utils import P4


package_path = str(pathlib.Path(__file__).parent.parent.resolve())


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
try:
    with gzip.open(package_path + "/data/jec_compiled.pkl.gz", "rb") as filehandler:
        jmestuff = pickle.load(filehandler)

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


def get_jec_jets(events: NanoEventsArray, year: str) -> FatJetArray:
    """
    Based on https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    Eventually update to V5 JECs once I figure out what's going on with the 2017 UL V5 JER scale factors
    """

    # fatjet_factory.build gies an error if there are no fatjets in event
    if not ak.any(events.FatJet):
        return events.FatJet

    import cachetools

    jec_cache = cachetools.Cache(np.inf)

    corr_key = f"{get_vfp_year(year)}mc"

    fatjets = fatjet_factory[corr_key].build(
        _add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache
    )

    return fatjets


# giving up on doing these myself for now because there's no 2017 UL V5 JER scale factors ???
# https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/JME_fatJet_jerc_Run2_UL/
# jec_stack_names = [
#     "Summer19UL17_V5_MC_L1FastJet_AK8PFPuppi",
#     "Summer19UL17_V5_MC_L2Relative_AK8PFPuppi",
#     "Summer19UL17_V5_MC_L3Absolute_AK8PFPuppi",
#     "Summer19UL17_V5_MC_L2L3Residual_AK8PFPuppi",
# ]


# jet definitions for LP SFs
dR = 0.8
cadef = fastjet.JetDefinition(fastjet.cambridge_algorithm, dR)
ktdef = fastjet.JetDefinition(fastjet.kt_algorithm, dR)
n_LP_sf_toys = 100


def _get_lund_arrays(events: NanoEventsArray, fatjet_idx: Tuple[int, ak.Array], num_prongs: int):
    """
    Gets the ``num_prongs`` subjet pTs and Delta and kT per primary LP splitting of fatjets at
    ``fatjet_idx`` in each event.

    Features are flattened (for now), and offsets are saved in ``ld_offsets`` to recover the event
    structure.

    Args:
        events (NanoEventsArray): nano events
        fatjet_idx (int | ak.Array): fatjet index
        num_prongs (int): number of prongs / subjets per jet to reweight

    Returns:
        flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, kt_subjets_vec
    """

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

    # save subjet pT
    kt_subjets_pt = kt_subjets_vec.pt
    # get constituents
    kt_subjet_consts = kt_clustering.exclusive_jets_constituents(num_prongs)

    # then re-cluster with CA
    # won't need to flatten once https://github.com/scikit-hep/fastjet/pull/145 is released
    ca_clustering = fastjet.ClusterSequence(ak.flatten(kt_subjet_consts, axis=1), cadef)
    lds = ak.flatten(ca_clustering.exclusive_jets_lund_declusterings(1), axis=1)

    # flatten and save offsets to unflatten afterwards
    ld_offsets = lds.kt.layout.offsets
    flat_logD = np.log(0.8 / ak.flatten(lds).Delta).to_numpy()
    flat_logkt = np.log(ak.flatten(lds).kt).to_numpy()
    # repeat subjet pt for each lund declustering
    flat_subjet_pt = np.repeat(ak.flatten(kt_subjets_pt), ak.count(lds.kt, axis=1)).to_numpy()

    return flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, kt_subjets_vec


def _calc_lund_SFs(
    flat_logD: np.ndarray,
    flat_logkt: np.ndarray,
    flat_subjet_pt: np.ndarray,
    ld_offsets: ak.Array,
    num_prongs: int,
    ratio_lookups: List[dense_lookup],
) -> np.ndarray:
    """
    Calculates scale factors for jets based on splittings in the primary Lund Plane.

    Lookup tables should be binned in [subjet_pt, ln(0.8/Delta), ln(kT/GeV)].

    Returns nominal scale factors for each lookup table in the ``ratio_smeared_lookups`` list.

    Args:
        flat_logD, flat_logkt, flat_subjet_pt, ld_offsets: numpy arrays from the ``lund_arrays`` fn
        num_prongs (int): number of prongs / subjets per jet to reweight
        ratio_smeared_lookups (List[dense_lookup]): list of lookup tables with smeared values

    Returns:
        nd.ndarray: SF values per jet for each smearing, shape ``[n_jets, len(ratio_lookups)]``.
    """

    sf_vals = []
    # could be parallelised but not sure if memory / time trade-off is worth it
    for i, ratio_lookup in enumerate(ratio_lookups):
        ratio_vals = ratio_lookup(flat_subjet_pt, flat_logD, flat_logkt)
        # recover jagged event structure
        reshaped_ratio_vals = ak.Array(
            ak.layout.ListOffsetArray64(ld_offsets, ak.layout.NumpyArray(ratio_vals))
        )
        # nominal values are product of all lund plane SFs
        sf_vals.append(
            # multiply subjet SFs per jet
            np.prod(
                # per-subjet SF
                ak.prod(reshaped_ratio_vals, axis=1).to_numpy().reshape(-1, num_prongs),
                axis=1,
            )
        )

    return np.array(sf_vals).T  # output shape: ``[n_jets, len(ratio_lookups)]``


def _get_lund_lookups(seed: int = 42, lnN: bool = True, trunc_gauss: bool = False):
    import uproot

    # initialize lund plane scale factors lookups
    f = uproot.open(package_path + "/corrections/lp_ratio_jan20.root")

    # 3D histogram: [subjet_pt, ln(0.8/Delta), ln(kT/GeV)]
    ratio_nom = f["ratio_nom"].to_numpy()
    ratio_nom_errs = f["ratio_nom"].errors()
    ratio_edges = ratio_nom[1:]
    ratio_nom = ratio_nom[0]

    ratio_sys_up = dense_lookup(f["ratio_sys_tot_up"].to_numpy()[0], ratio_edges)
    ratio_sys_down = dense_lookup(f["ratio_sys_tot_down"].to_numpy()[0], ratio_edges)

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

        kappa = (ratio_nom + ratio_nom_errs) / ratio_nom
        ratio_nom_smeared = ratio_nom * np.power(kappa, rand_noise)
        ratio_lnN_smeared_lookups = [dense_lookup(ratio_nom, ratio_edges)] + [
            dense_lookup(ratio_nom_smeared[i], ratio_edges) for i in range(n_LP_sf_toys)
        ]
    else:
        ratio_lnN_smeared_lookups = None

    return ratio_smeared_lookups, ratio_lnN_smeared_lookups, ratio_sys_up, ratio_sys_down


(
    ratio_smeared_lookups,
    ratio_lnN_smeared_lookups,
    ratio_sys_up,
    ratio_sys_down,
) = (
    None,
    None,
    None,
    None,
)


def get_lund_SFs(
    events: NanoEventsArray,
    fatjet_idx: Tuple[int, ak.Array],
    num_prongs: int,
    gen_quarks: GenParticleArray,
    seed: int = 42,
    trunc_gauss: bool = False,
    lnN: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Calculates scale factors for jets based on splittings in the primary Lund Plane.
    Calculates random smearings for statistical uncertainties, total up/down systematic variation,
    and subjet matching and pT extrapolation systematic uncertainties.

    Args:
        events (NanoEventsArray): nano events
        fatjet_idx (int | ak.Array): fatjet index
        num_prongs (int): number of prongs / subjets per jet to r
        seed (int, optional): seed for random smearings. Defaults to 42.
        trunc_gauss (bool, optional): use truncated gaussians for smearing. Defaults to False.
        lnN (bool, optional): use log normals for smearings. Defaults to True.

    Returns:
        Dict[str, np.ndarray]: dictionary with nominal weights per jet, sys variations, and (optionally) random smearings.
    """

    # global variable to not have to load + smear LP ratios each time
    global ratio_smeared_lookups, ratio_lnN_smeared_lookups, ratio_sys_up, ratio_sys_down

    if (lnN and ratio_lnN_smeared_lookups is None) or (
        trunc_gauss and ratio_smeared_lookups is None
    ):
        (
            ratio_smeared_lookups,
            ratio_lnN_smeared_lookups,
            ratio_sys_up,
            ratio_sys_down,
        ) = _get_lund_lookups(seed, lnN, trunc_gauss)

    flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, kt_subjets_vec = _get_lund_arrays(
        events, fatjet_idx, num_prongs
    )

    sfs = {}

    ### get scale factors per jet + smearings for stat unc. + syst. variations

    if trunc_gauss:
        sfs["lp_sf"] = _calc_lund_SFs(
            flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, num_prongs, ratio_smeared_lookups
        )

    if lnN:
        sfs["lp_sf_lnN"] = _calc_lund_SFs(
            flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, num_prongs, ratio_lnN_smeared_lookups
        )

    sfs["lp_sf_sys_down"] = _calc_lund_SFs(
        flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, num_prongs, [ratio_sys_down]
    )

    sfs["lp_sf_sys_up"] = _calc_lund_SFs(
        flat_logD, flat_logkt, flat_subjet_pt, ld_offsets, num_prongs, [ratio_sys_up]
    )

    ### subjet matching and pT extrapolation uncertainties

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

    # events which have more than one quark matched to the same subjet
    sfs["lp_sf_double_matched_event"] = np.any(
        [np.sum(sj_matched_idx_mask == i, axis=1) > 1 for i in range(3)], axis=0
    ).astype(int)[:, np.newaxis]

    # number of quarks per event which aren't matched
    sfs["lp_sf_unmatched_quarks"] = np.sum(~sj_matched, axis=1, keepdims=True)

    # pT extrapolation uncertainty
    sfs["lp_sf_num_sjpt_gt350"] = np.sum(kt_subjets_vec.pt > 350, axis=1, keepdims=True).to_numpy()

    return sfs
